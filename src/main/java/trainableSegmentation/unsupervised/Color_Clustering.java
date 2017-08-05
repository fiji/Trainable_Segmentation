package trainableSegmentation.unsupervised;


import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.ImageCanvas;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.StackWindow;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.plugin.PlugIn;
import trainableSegmentation.FeatureStackArray;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;
import weka.gui.explorer.ClustererAssignmentsPlotInstances;

import weka.gui.visualize.VisualizePanel;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import java.awt.BorderLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Label;
import java.awt.Panel;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Fiji plugin to perform color clustering on a 2D image
 * or stack based on different color space representations
 * (RGB, HSB and Lab) and all available clusterers in Weka.
 *
 * @author Josu Salinas and Ignacio Arganda-Carreras
 *
 */
public class Color_Clustering implements PlugIn{

    //GUI reestructurar; crear script para probar;

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
    private boolean featuresCreated = false;
    private FeatureStackArray theFeatures = null;
    private boolean clustererLoaded = false;


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
        private JButton createResult = null;
        private JButton createProbabilityMap = null;
        private JButton visualizeData = null;
        private JButton saveClusterer = null;
        private JButton loadClusterer = null;
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
                if(featuresCreated){
                    featuresCreated =false;
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
         * Action listener for action performed
         */
        ActionListener saveTheClusterer = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        saveClusterer();
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
                        updateClusterer();
                        if(featuresCreated){
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
         * Action listener for result creation
         */
        ActionListener resultCreation = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        ImagePlus result = clusteredImage.duplicate();
                        result.setCalibration(image.getCalibration());
                        result.setFileInfo(image.getFileInfo());
                        result.setTitle(image.getShortTitle()+"clusters");
                        result.show();
                    }
                });
            }
        };

        /**
         * Action listener for probability map creation
         */
        ActionListener probMapCreator = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        createFeatures();
                        if(clusterer==null||!clustererLoaded){
                            updateClusterer();
                        }
                        AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                        colorClustering.setTheClusterer(theClusterer);
                        IJ.log(theClusterer.toString());
                        ImagePlus result = colorClustering.createProbabilityMaps(theFeatures);
                        result.setCalibration(image.getCalibration());
                        result.setFileInfo(image.getFileInfo());
                        result.setTitle(image.getShortTitle()+"clusterprobmap");
                        result.show();
                    }
                });
            }
        };

        /**
         * Action listener for data visualization
         */
        ActionListener dataVisualizer = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        visualiseData();
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

        /**
         * Action listener for loading clusterers
         */
        ActionListener clusterLoader = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                loadClusterer();
            }
        };

        /**
         * Action lsitener for channel selection, sets featuresCreated flag as false in order to force feature creation on next use.
         */
        ActionListener channelSelect = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                exec.submit(new Runnable() {
                    public void run() {
                        featuresCreated =false;
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
            colorClustering = new ColorClustering(image);
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            numChannels = ColorClustering.Channel.numChannels();
            String[] channelList = ColorClustering.Channel.getAllLabels();
            for(int i=0;i<numChannels;++i){
                JCheckBox tmp = new JCheckBox(channelList[i]);
                tmp.addActionListener(channelSelect);
               channelSelection.add(tmp,i);
            }

            int height = image.getHeight();
            int width = image.getWidth();

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
                    /*super.zSelector.addAdjustmentListener(new AdjustmentListener() {
                        @Override
                        public void adjustmentValueChanged(AdjustmentEvent e) {
                            if( overlayEnabled )
                            {
                                updateResultOverlay();
                                displayImage.updateAndDraw();

                            }
                            IJ.log("Test1");
                        }
                    });*/
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
            samplePanel.setBorder(BorderFactory.createTitledBorder("Number of Samples"));
            samplePanel.setToolTipText("Select a percentage of pixels to be used when training the clusterer");
            samplePanel.add(pixelSlider,1);
            JTextArea txtNumSamples = null;
            if(((image.getHeight()*image.getWidth()*image.getNSlices()) * 0.5) < 30000){
                txtNumSamples = new JTextArea("50% ("+Integer.toString(((image.getHeight()*image.getWidth()*image.getNSlices()) * pixelSlider.getValue() / 100))+") px");
            }else{
                double x = (30000 / (double) (image.getHeight()*image.getWidth()*image.getNSlices()));
                if(x<0.01){
                    x=0.01;
                }
                txtNumSamples = new JTextArea(Integer.toString( (int) (x*100))+"% ("+Integer.toString(((image.getHeight()*image.getWidth()*image.getNSlices()) * ((int) (x*100))/100))+") px");
                pixelSlider.setValue( (int) (x*100));
            }
            numSamples=((image.getHeight()*image.getWidth())*image.getNSlices()) * pixelSlider.getValue() / 100;
            samplePanel.add(txtNumSamples,2);
            pixelSlider.addChangeListener(sampleChange);
            all.add(samplePanel,allConstraints);
            allConstraints.gridy++;

            clusterer = new SimpleKMeans();
            PropertyPanel clustererEditorPanel = new PropertyPanel( clustererEditor );
            clustererEditor.setClassType( Clusterer.class );
            clustererEditor.setValue( clusterer );
            clustererEditor.addPropertyChangeListener(new PropertyChangeListener() {
                @Override
                public void propertyChange(PropertyChangeEvent evt) {
                    updateClusterer();
                }
            });
            clusterizerSelection.add(clustererEditorPanel);
            clusterizerSelection.setBorder(BorderFactory.createTitledBorder("Clusterer"));
            clusterizerSelection.setToolTipText("Choose clusterer to be used");
            all.add(clusterizerSelection,allConstraints);
            allConstraints.gridy++;

            clusterizeButton = new JButton("Clusterize");
            executor.add(clusterizeButton);
            clusterizeButton.setToolTipText("Clusterize the image!");

            clusterizeButton.addActionListener(clusterize);

            createFile = new JButton("Create ARFF file");
            createFile.setToolTipText("Create a file");
            createFile.addActionListener(fileCreation);
            executor.add(createFile);

            toggleOverlay = new JButton("Toggle overlay");
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

            createResult = new JButton("Show result");
            executor.add(createResult);
            createResult.addActionListener(resultCreation);
            createResult.setEnabled(false);

            createProbabilityMap = new JButton("Probability Map");
            executor.add(createProbabilityMap);
            createProbabilityMap.addActionListener(probMapCreator);
            createProbabilityMap.setEnabled(false);

            visualizeData = new JButton("Visualize data");
            executor.add(visualizeData);
            visualizeData.addActionListener(dataVisualizer);

            saveClusterer = new JButton("Save clusterer");
            executor.add(saveClusterer);
            saveClusterer.addActionListener(saveTheClusterer);
            saveClusterer.setEnabled(false);

            loadClusterer = new JButton("Load clusterer");
            executor.add(loadClusterer);
            loadClusterer.addActionListener(clusterLoader);

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

        /**
         * Creates features based on selected channels and number of samples (from GUI)
         * @return succes or failure
         */
        boolean createFeatures(){
            boolean someChannelSelected = false;

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
                featuresCreated = true;
                return true;
            } else {
                IJ.error("Warning!","Choose at least a channel");
                clusterizeButton.setText("Clusterize");
                return false;
            }
        }

        /**
         * Creates featres and displays the data
         */
        void visualiseData(){
            if(!featuresCreated){
                if(!createFeatures()){
                    return;
                }
            }
            Instances train = colorClustering.getFeaturesInstances();
            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(clusterer);
            try {
                eval.evaluateClusterer(train);
            } catch (Exception e) {
                e.printStackTrace();
            }
            ClustererAssignmentsPlotInstances plotInstances = new ClustererAssignmentsPlotInstances();
            plotInstances.setClusterer(clusterer);
            plotInstances.setInstances(train);
            plotInstances.setClusterEvaluation(eval);
            plotInstances.setUp();
            String name = (new SimpleDateFormat("HH:mm:ss - ")).format(new Date());
            String cname = clusterer.getClass().getName();
            if (cname.startsWith("weka.clusterers."))
                name += cname.substring("weka.clusterers.".length());
            else
                name += cname;
            name = name + " (" + train.relationName() + ")";
            VisualizePanel vp = new VisualizePanel();
            vp.setName(name);
            try {
                vp.addPlot(plotInstances.getPlotData(cname));
            } catch (Exception e) {
                e.printStackTrace();
            }

            // display data
            // taken from: ClustererPanel.visualizeClusterAssignments(VisualizePanel)
            JFrame jf = new JFrame("Weka Clusterer Visualize: " + vp.getName());
            jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            jf.setSize(500, 400);
            jf.getContentPane().setLayout(new BorderLayout());
            jf.getContentPane().add(vp, BorderLayout.CENTER);
            jf.setVisible(true);
        }


        /**
         * Clusterizes the selected image if the command is "Clusterize", else it stops the custerization.
         * WARNING: Weka does not support usage of interrupt flag and instead uses deprecated stop function.
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
                        clusterer = colorClustering.getTheClusterer();
                        if(!clustererLoaded){
                            updateClusterer();
                        }
                        if(featuresCreated&&!clustererLoaded){
                            AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                            colorClustering.setTheClusterer(theClusterer);
                            IJ.log(theClusterer.toString());
                            clusteredImage = colorClustering.createClusteredImage(theFeatures);
                            overlayEnabled=true;
                            updateResultOverlay();
                            clusterizeButton.setText("Clusterize");
                            if (!toggleOverlay.isEnabled()) {
                                toggleOverlay.setEnabled(true);
                                opacitySlider.setEnabled(true);
                                saveClusterer.setEnabled(true);
                                createProbabilityMap.setEnabled(true);

                            }
                        }else {
                            if(createFeatures()) {
                                AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
                                colorClustering.setTheClusterer(theClusterer);
                                IJ.log(theClusterer.toString());
                                clusteredImage = colorClustering.createClusteredImage(theFeatures);
                                overlayEnabled=true;
                                updateResultOverlay();
                                clusterizeButton.setText("Clusterize");
                                if (!toggleOverlay.isEnabled()) {
                                    toggleOverlay.setEnabled(true);
                                    opacitySlider.setEnabled(true);
                                    saveClusterer.setEnabled(true);
                                    createProbabilityMap.setEnabled(true);
                                    createResult.setEnabled(true);
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

        /**
         * Updates clusterer based on selected options on GUI
         */
        public void updateClusterer(){
            if(clusterer!=null) {
                String[] prevOptions = clusterer.getOptions();
                Object c = (Object) clustererEditor.getValue();
                String options = "";
                String[] optionsArray = ((OptionHandler) c).getOptions();
                if (c instanceof OptionHandler)
                {
                    options = Utils.joinOptions(optionsArray);
                }

                if (optionsArray.length != prevOptions.length) {
                    clustererLoaded = false;
                    try {
                        clusterer = (AbstractClusterer) (c.getClass().newInstance());
                        clusterer.setOptions(optionsArray);
                    } catch (
                            Exception ex)

                    {
                        clusterizeButton.setText("Clusterize");
                        IJ.log("Error when setting clusterer");
                    }

                } else {
                    for (int i = 0; i < optionsArray.length; ++i) {
                        if (!prevOptions[i].contentEquals(optionsArray[i])) {
                            clustererLoaded = false;
                            try {
                                clusterer = (AbstractClusterer) (c.getClass().newInstance());
                                clusterer.setOptions(optionsArray);
                            } catch (
                                    Exception ex)

                            {
                                clusterizeButton.setText("Clusterize");
                                IJ.log("Error when setting clusterer");
                            }
                            break;
                        }
                    }
                }
            }else{
                clustererLoaded=false;
                Object c = (Object) clustererEditor.getValue();
                String options = "";
                String[] optionsArray = ((OptionHandler) c).getOptions();
                if (c instanceof OptionHandler)

                {
                    options = Utils.joinOptions(optionsArray);
                }
                try {
                    clusterer = (AbstractClusterer) (c.getClass().newInstance());
                    clusterer.setOptions(optionsArray);
                } catch (
                        Exception ex)

                {
                    clusterizeButton.setText("Clusterize");
                    IJ.log("Error when setting clusterer");
                }
            }
        }
    }

    /**
     * Save the clusterer to a file
     */
    public void saveClusterer(){
        SaveDialog sd = new SaveDialog("Save model as...", "clusterer",".model");
        if (sd.getFileName()==null)
            return;

        if( !colorClustering.saveClusterer(sd.getDirectory() + sd.getFileName()) )
        {
            IJ.error("Error while writing clusterer into a file");
            return;
        }
    }

    /**
     * Load clusterer from a file and set selected channels
     */
    public void loadClusterer(){
        OpenDialog od = new OpenDialog( "Choose Weka clusterer file", "" );
        if (od.getFileName()==null)
            return;
        IJ.log("Loading Weka clusterer from " + od.getDirectory() + od.getFileName() + "...");

        if(  !colorClustering.loadClusterer(od.getDirectory() + od.getFileName()) )
        {
            IJ.error("Error when loading Weka clusterer from file");
            IJ.log("Error: clusterer could not be loaded.");
            return;
        }else {
            clustererLoaded=true;
            clusterer=colorClustering.getTheClusterer();
            Instances featuresInstances = colorClustering.getFeaturesInstances();
            Boolean[] enabledChannels = new Boolean[numChannels];
            for(int i=0;i<numChannels;++i){
                enabledChannels[i]=false;
            }
            for(int i=0;i<featuresInstances.numAttributes();++i){
                String name = featuresInstances.attribute(i).name();
                switch (name){
                    case "Red":
                        enabledChannels[0]=true;
                        break;
                    case "Green":
                        enabledChannels[1]=true;
                        break;
                    case "Blue":
                        enabledChannels[2]=true;
                        break;
                    case "L":
                        enabledChannels[3]=true;
                        break;
                    case "a":
                        enabledChannels[4]=true;
                        break;
                    case "b":
                        enabledChannels[5]=true;
                        break;
                    case "Hue":
                        enabledChannels[6]=true;
                        break;
                    case "Saturation":
                        enabledChannels[7]=true;
                        break;
                    case "Brightness":
                        enabledChannels[8]=true;
                        break;
                }
            }
            for(int i=0;i<numChannels;++i){
                JCheckBox checkBox = (JCheckBox) win.channelSelection.getComponent(i);
                checkBox.setSelected(enabledChannels[i]);
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
        if(image == null){
            IJ.error("Error when opening image");
        }else {
            IJ.log("Loading Weka properties");
            win = new CustomWindow(image);
        }

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
        IJ.runPlugIn(clazz.getName(),"");

    }

}
