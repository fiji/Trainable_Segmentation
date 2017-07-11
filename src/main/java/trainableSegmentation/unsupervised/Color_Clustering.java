package trainableSegmentation.unsupervised;


import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.*;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import javafx.scene.control.CheckBox;
import javafx.scene.control.RadioButton;
import jdk.nashorn.internal.runtime.arrays.NumericElements;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.Weka_Segmentation;
import vib.segment.Border;
import vib.segment.CustomCanvas;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Check;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.stopwords.Null;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.*;
import java.nio.channels.Channel;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Color_Clustering implements PlugIn{


    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    protected ImagePlus image=null;
    private boolean[] selectedChannels;
    private int numSamples;
    private int numChannels;
    private boolean file=false;
    private AbstractClusterer clusterer;
    ImagePlus displayImage = null;
    private Thread currentTask=null;
    private ImagePlus clusteredImage=null;
    private CustomWindow win;
    private boolean overlayEnabled = false;

    //Morph segm, window closing, check overlay and title

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
        private boolean warned=false;
        private JSlider slider;

        ChangeListener sampleChange = new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                if(!warned&&numSamples>1000000){
                    IJ.error("Warning","Pixel count very high!");
                    warned=true;
                }else if(warned&&numSamples<1000000){
                    warned=false;
                }
                JSlider slider = (JSlider) samplePanel.getComponent(1);
                numSamples = ((image.getHeight()*image.getWidth())*image.getNSlices()) * slider.getValue() / 100;
                JTextArea textArea = (JTextArea) samplePanel.getComponent(2);
                textArea.setText(Integer.toString(slider.getValue())+"% ("+numSamples+") " + "px");
            }
        };

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

        ActionListener overlay = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        if(e.getSource()==toggleOverlay) {
                            if(overlayEnabled){
                                overlayEnabled=false;
                                image.setOverlay(null);
                                overlayEnabled=false;
                            }else {
                                updateResultOverlay();
                                overlayEnabled=true;
                            }
                        }
                    }
                });
            }
        };

        void updateResultOverlay()
        {
            if( null != clusteredImage )
            {
                overlayEnabled=true;
                int slice = image.getCurrentSlice();
                ImageRoi roi = null;
                roi = new ImageRoi(0, 0, clusteredImage.getImageStack().getProcessor(slice));
                roi.setOpacity(0.5);
                image.setOverlay(new Overlay(roi));
            }
        }

        CustomWindow(ImagePlus imp) {
            super(imp, new ImageCanvas(imp));
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            numChannels = ColorClustering.Channel.numChannels();
            String[] channelList = ColorClustering.Channel.getAllLabels();
            for(int i=0;i<numChannels;++i){
               channelSelection.add(new JCheckBox(channelList[i]),i);
            }
            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.NORTHWEST;
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
            if( null != super.sliceSelector )
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

            } //Add listener para cambiar overlay, como en morph (mouse, wheel key etc)

            samplePanel.add(new Label("Select sample percentage:"));
            slider = new JSlider(1,100,50);
            samplePanel.add(slider,1);
            samplePanel.setBorder(BorderFactory.createTitledBorder("Number of Samples"));
            samplePanel.setToolTipText("Select a percentage of pixels to be used when training the clusterer");
            JTextArea txtNumSamples = new JTextArea("50% ("+Integer.toString(((image.getHeight()*image.getWidth()) * slider.getValue() / 100))+") px");
            numSamples=((image.getHeight()*image.getWidth())*image.getNSlices()) * slider.getValue() / 100;
            samplePanel.add(txtNumSamples,2);
            slider.addChangeListener(sampleChange);
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

            toggleOverlay = new JButton("Overlay");
            executor.add(toggleOverlay);
            toggleOverlay.setToolTipText("Toggle result image overlay!");
            toggleOverlay.addActionListener(overlay);


            all.add(executor,allConstraints);


            GridBagLayout wingb = new GridBagLayout();
            GridBagConstraints winc = new GridBagConstraints();
            winc.anchor = GridBagConstraints.NORTHWEST;
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

        @Override
        public void windowClosing(WindowEvent e) {
            super.windowClosing(e);
            if(null != image){
                if(null != displayImage){
                    image.setSlice(displayImage.getCurrentSlice());
                }
                image.getWindow().setVisible(true);
            }
            clusterizeButton.removeActionListener(clusterize);
            slider.removeChangeListener(sampleChange);
            if(null != displayImage){
                displayImage=null;
            }
            exec.shutdownNow();
        }

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
                        //cambiar IJ.error
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
                            ColorClustering colorClustering = new ColorClustering(image, numSamples, channels);
                            AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                            colorClustering.setTheClusterer(theClusterer);
                            IJ.log(theClusterer.toString());
                            FeatureStackArray theFeatures = colorClustering.createFSArray(image);
                            clusteredImage = colorClustering.createClusteredImage(theFeatures);
                            clusteredImage.show();
                            clusterizeButton.setText("Clusterize");
                        } else

                        {
                            IJ.error("Warning!","Choose at least a channel");
                            clusterizeButton.setText("Clusterize");
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
