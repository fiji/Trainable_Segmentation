package trainableSegmentation.unsupervised;


import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.*;
import ij.plugin.PlugIn;
import trainableSegmentation.FeatureStackArray;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Color_Clustering implements PlugIn{

    //graficos; toggle overlay+show clusterized (copia); probability map; file open; GUI reestructurar; crear script para probar

    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    protected ImagePlus image=null;
    private boolean[] selectedChannels;
    private int numSamples;
    private int numChannels;
    private AbstractClusterer clusterer;
    ImagePlus displayImage = null;
    private Thread currentTask=null;
    private ImagePlus clusteredImage=null;
    private CustomWindow win;
    private boolean overlayEnabled = false;
    private ColorClustering colorClustering = null;
    private boolean instancesCreated = false;
    private FeatureStackArray theFeatures = null;


    /**
     * Custom window based on JPanel structures
     */
    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel channelSelection = new JPanel();
        private JPanel clusterizerSelection = new JPanel();
        private JPanel executor = new JPanel();
        private JPanel samplePanel = new JPanel();
        private GenericObjectEditor clustererEditor = new GenericObjectEditor();
        private JButton clusterizeButton = null;
        private JButton toggleOverlay = null;
        private JButton createFile = null;
        private boolean warned=false;
        private JSlider pixelSlider;
        private JSlider opacitySlider;

        /**
         * Change listener for sample pixel count
         */
        ChangeListener sampleChange = new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                if(!warned&&numSamples>1000000){
                    IJ.error("Warning","Pixel count very high!");
                    warned=true;
                }else if(warned&&numSamples<1000000){
                    warned=false;
                }
                if(instancesCreated){
                    instancesCreated=false;
                }
                JSlider slider = (JSlider) samplePanel.getComponent(1);
                numSamples = ((image.getHeight()*image.getWidth())*image.getNSlices()) * slider.getValue() / 100;
                JTextArea textArea = (JTextArea) samplePanel.getComponent(2);
                textArea.setText(Integer.toString(slider.getValue())+"% ("+numSamples+") " + "px");
            }
        };
        /**
         * Change listener for overlay opacity percentage
         */
        ChangeListener opacityChange = new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                updateResultOverlay();
            }
        };

        /**
         * Action listener for clusterize button
         */
        ActionListener clusterize = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        if(e.getSource()==clusterizeButton) {
                            clusterizeOrStop(command);
                        }
                    }
                });
            }
        };


        /**
         * Action listener for file creation button
         */
        ActionListener fileCreation = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        if(instancesCreated){
                            colorClustering.createFile(colorClustering.getFeaturesInstances());
                        }else {
                            if(createFeatures()) {
                                colorClustering.createFile(colorClustering.getFeaturesInstances());
                            }
                        }
                    }
                });
            }
        };

        /**
         * Action listener for overlay button
         */
        ActionListener overlay = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        if(e.getSource()==toggleOverlay) {
                            if(overlayEnabled){
                                opacitySlider.setEnabled(false);
                                overlayEnabled=false;
                                image.setOverlay(null);
                                overlayEnabled=false;
                            }else{
                                updateResultOverlay();
                                opacitySlider.setEnabled(true);
                                overlayEnabled=true;
                            }
                        }
                    }
                });
            }
        };

        ActionListener channelSelect = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                exec.submit(new Runnable() {
                    public void run() {
                        instancesCreated=false;
                    }
                });
            }
        };

        /**
         * Update the overlay
         */
        void updateResultOverlay()
        {
            if( null != clusteredImage )
            {
                overlayEnabled=true;
                int slice = image.getCurrentSlice();
                ImageRoi roi = null;
                roi = new ImageRoi(0, 0, clusteredImage.getImageStack().getProcessor(slice));
                roi.setOpacity((double) opacitySlider.getValue()/100);
                image.setOverlay(new Overlay(roi));
            }
        }

        /**
         * Custom window creator
         * @param imp
         */
        CustomWindow(ImagePlus imp) {
            super(imp, new ImageCanvas(imp));
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            numChannels = ColorClustering.Channel.numChannels();
            String[] channelList = ColorClustering.Channel.getAllLabels();
            for(int i=0;i<numChannels;++i){
                JCheckBox tmp = new JCheckBox(channelList[i]);
                tmp.addActionListener(channelSelect);
               channelSelection.add(tmp,i);
            }

            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.CENTER;
            allConstraints.fill = GridBagConstraints.BOTH;
            allConstraints.gridwidth = 1;
            allConstraints.gridheight = 1;
            allConstraints.gridx = 0;
            allConstraints.gridy = 0;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            channelSelection.setBorder(BorderFactory.createTitledBorder("Channel"));
            channelSelection.setToolTipText("Choose channels to be used");
            all.add(channelSelection,allConstraints);

            allConstraints.gridy++;
            all.add(canvas,allConstraints);
            allConstraints.gridy++;
            // if the input image is 3d, put the
            // slice selectors in place
            if( null != super.sliceSelector ) //Adjustment listener añadir a tSelector
            {
                sliceSelector.setValue( image.getCurrentSlice() );
                image.setSlice( image.getCurrentSlice() );

                all.add( super.sliceSelector, allConstraints );
                allConstraints.gridy++;
                if( null != super.zSelector ) {
                    all.add(super.zSelector, allConstraints);
                    allConstraints.gridy++;
                }
                if( null != super.tSelector ){
                    all.add( super.tSelector, allConstraints );
                    allConstraints.gridy++;
                }
                if( null != super.cSelector ){
                    all.add( super.cSelector, allConstraints );
                    allConstraints.gridy++;
                }

            }//Add listener para cambiar overlay, como en morph (mouse, wheel key etc)

            samplePanel.add(new Label("Select sample percentage:"));
            pixelSlider = new JSlider(1,100,50);
            samplePanel.add(pixelSlider,1);
            samplePanel.setBorder(BorderFactory.createTitledBorder("Number of Samples"));
            samplePanel.setToolTipText("Select a percentage of pixels to be used when training the clusterer");
            JTextArea txtNumSamples = new JTextArea("50% ("+Integer.toString(((image.getHeight()*image.getWidth()) * pixelSlider.getValue() / 100))+") px");
            numSamples=((image.getHeight()*image.getWidth())*image.getNSlices()) * pixelSlider.getValue() / 100;
            samplePanel.add(txtNumSamples,2);
            pixelSlider.addChangeListener(sampleChange);
            all.add(samplePanel,allConstraints);
            allConstraints.gridy++;

            clusterer = new SimpleKMeans();
            PropertyPanel clustererEditorPanel = new PropertyPanel( clustererEditor );
            clustererEditor.setClassType( Clusterer.class );
            clustererEditor.setValue( clusterer );
            clusterizerSelection.add(clustererEditorPanel);
            clusterizerSelection.setBorder(BorderFactory.createTitledBorder("Clusterer"));
            clusterizerSelection.setToolTipText("Choose clusterer to be used");
            all.add(clusterizerSelection,allConstraints);
            allConstraints.gridy++;

            clusterizeButton = new JButton("Clusterize");
            executor.add(clusterizeButton);
            clusterizeButton.setToolTipText("Clusterize the image!");

            clusterizeButton.addActionListener(clusterize);

            createFile = new JButton("Create File");
            createFile.setToolTipText("Create a file");
            createFile.addActionListener(fileCreation);
            executor.add(createFile);

            toggleOverlay = new JButton("Overlay");
            executor.add(toggleOverlay);
            toggleOverlay.setToolTipText("Toggle result image overlay!");
            toggleOverlay.addActionListener(overlay);
            toggleOverlay.setEnabled(false);

            opacitySlider = new JSlider(0,100,50);
            executor.add(new Label("Select overlay opacity:"));
            opacitySlider.setToolTipText("Select a percentage for the opacity");
            executor.add(opacitySlider);
            opacitySlider.addChangeListener(opacityChange);
            opacitySlider.setEnabled(false);

            all.add(executor,allConstraints);


            GridBagLayout wingb = new GridBagLayout();
            GridBagConstraints winc = new GridBagConstraints();
            winc.anchor = GridBagConstraints.CENTER;
            winc.fill = GridBagConstraints.BOTH;
            winc.weightx = 0;
            winc.weighty = 0;
            setLayout( wingb );
            add( all, winc );

            // Fix minimum size to the preferred size at this point
            pack();
            setMinimumSize( getPreferredSize() );

            if(null != sliceSelector)
            {
                // add adjustment listener to the scroll bar
                sliceSelector.addAdjustmentListener(new AdjustmentListener()
                {

                    public void adjustmentValueChanged(final AdjustmentEvent e) {
                        exec.submit(new Runnable() {
                            public void run() {
                                if(e.getSource() == sliceSelector)
                                {
                                    if( overlayEnabled )
                                    {
                                        updateResultOverlay();
                                        displayImage.updateAndDraw();

                                    }
                                }

                            }
                        });
                    }
                });

                // mouse wheel listener to update the rois while scrolling
                addMouseWheelListener(new MouseWheelListener() {

                    @Override
                    public void mouseWheelMoved(final MouseWheelEvent e) {

                        exec.submit(new Runnable() {
                            public void run()
                            {
                                if( overlayEnabled )
                                {
                                    updateResultOverlay();
                                    displayImage.updateAndDraw();
                                }
                            }
                        });

                    }
                });

                // key listener to repaint the display image and the traces
                // when using the keys to scroll the stack
                KeyListener keyListener = new KeyListener() {

                    @Override
                    public void keyTyped(KeyEvent e) {}

                    @Override
                    public void keyReleased(final KeyEvent e) {
                        exec.submit(new Runnable() {
                            public void run()
                            {
                                if(e.getKeyCode() == KeyEvent.VK_LEFT ||
                                        e.getKeyCode() == KeyEvent.VK_RIGHT ||
                                        e.getKeyCode() == KeyEvent.VK_LESS ||
                                        e.getKeyCode() == KeyEvent.VK_GREATER ||
                                        e.getKeyCode() == KeyEvent.VK_COMMA ||
                                        e.getKeyCode() == KeyEvent.VK_PERIOD)
                                {
                                    if( overlayEnabled )
                                    {
                                        updateResultOverlay();
                                        displayImage.updateAndDraw();
                                    }
                                }
                            }
                        });

                    }

                    @Override
                    public void keyPressed(KeyEvent e) {}
                };
                // add key listener to the window and the canvas
                addKeyListener(keyListener);
                canvas.addKeyListener(keyListener);

            }

        }

        boolean createFeatures(){
            boolean someChannelSelected = false;
            Object c = (Object) clustererEditor.getValue();
            String options = "";
            String[] optionsArray = ((OptionHandler) c).getOptions();
            if (c instanceof OptionHandler)

            {
                options = Utils.joinOptions(optionsArray);
            }
            try

            {
                clusterer = (AbstractClusterer) (c.getClass().newInstance());
                clusterer.setOptions(optionsArray);
            } catch (
                    Exception ex)

            {
                clusterizeButton.setText("Clusterize");
                IJ.log("Error when setting clusterer");
            }

            selectedChannels = new boolean[numChannels];
            numChannels = ColorClustering.Channel.numChannels();
            for (
                    int i = 0;
                    i < numChannels; ++i)

            {
                JCheckBox selected = (JCheckBox) channelSelection.getComponent(i);
                selectedChannels[i] = selected.isSelected();
                if (selected.isSelected() && !someChannelSelected) {
                    someChannelSelected = true;
                }
            }
            if (someChannelSelected)
            {
                IJ.log("Number of selected samples: " + numSamples);
                ArrayList<ColorClustering.Channel> channels = new ArrayList<ColorClustering.Channel>();
                for (int i = 0; i < numChannels; ++i) {
                    if (selectedChannels[i]) {
                        ColorClustering.Channel channel = ColorClustering.Channel.fromLabel(ColorClustering.Channel.getAllLabels()[i]);
                        channels.add(channel);
                    }
                }
                colorClustering = new ColorClustering(image, numSamples, channels);
                IJ.log("Creating features");
                theFeatures = colorClustering.createFSArray(image);
                instancesCreated = true;
                return true;
            } else {
                IJ.error("Warning!","Choose at least a channel");
                clusterizeButton.setText("Clusterize");
                return false;
            }
        }


        /**
         * Based on command clusterize image or stop clusterization procedure
         * @param command
         */
        void clusterizeOrStop(String command){
            IJ.log("Command: "+command);
            if(command.equals("Clusterize")){
                clusterizeButton.setText("STOP");
                final Thread oldTask = currentTask;
                Thread newTask = new Thread() {

                    public void run() {

                        if (null != oldTask)
                        {
                            try {
                                IJ.log("Waiting for old task to finish...");
                                oldTask.join();
                            }
                            catch (InterruptedException ie)	{ IJ.log("interrupted"); }
                        }
                        if(instancesCreated){
                            AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                            colorClustering.setTheClusterer(theClusterer);
                            IJ.log(theClusterer.toString());
                            clusteredImage = colorClustering.createClusteredImage(theFeatures);
                            clusteredImage.show();
                            clusterizeButton.setText("Clusterize");
                            if (!toggleOverlay.isEnabled()) {
                                toggleOverlay.setEnabled(true);
                            }
                        }else {
                            if(createFeatures()) {
                                AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                                colorClustering.setTheClusterer(theClusterer);
                                IJ.log(theClusterer.toString());
                                clusteredImage = colorClustering.createClusteredImage(theFeatures);
                                clusteredImage.show();
                                clusterizeButton.setText("Clusterize");
                                if (!toggleOverlay.isEnabled()) {
                                    toggleOverlay.setEnabled(true);
                                }
                            }
                        }
                    }
                };
                currentTask = newTask;
                newTask.start();
            }else if(command.equals("STOP")){
                IJ.log("Clusterization stopped by user");
                clusterizeButton.setText("Clusterize");
                if(null != currentTask) {
                    currentTask.interrupt();//Should use interrupt but weka does not support interrupt handling.
                    currentTask.stop();//Interrupt is being used
                }else{
                    IJ.log("Error: Interrupting failed because thread was null");
                }
            }
        }

    }


    /**
     * Run function for plug-in
     * @param s
     */
    @Override
    public void run(String s) {

        image = WindowManager.getCurrentImage();
        if(image == null){
            image=IJ.openImage();
        }
        IJ.log("Loading Weka properties");
        win = new CustomWindow(image);

    }

    /**
     * Usefull for testing
     * @param args
     */
    public static void main(String[] args){
        Class<?> clazz = Color_Clustering.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
        System.setProperty("plugins.dir", pluginsDir);
        new ImageJ();
        ImagePlus image = IJ.openImage();
        image.show();
        IJ.runPlugIn(clazz.getName(),"");

    }

}
