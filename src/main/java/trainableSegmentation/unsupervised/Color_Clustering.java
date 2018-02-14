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
import java.awt.GridLayout;
import java.awt.Insets;
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
    private Thread currentTask=null;
    private ImagePlus clusteredImage=null;
    private CustomWindow win;
    private boolean overlayEnabled = false;
    private ColorClustering colorClustering = null;
    private boolean featuresCreated = false;
    private FeatureStackArray theFeatures = null;
    private boolean clustererLoaded = false;
    /** input image title */
    String inputImageTitle = null;
    /** input image short title */
    String inputImageShortTitle = null;

    /**
     * Custom window based on JPanel structures
     */
    private class CustomWindow extends StackWindow
    {
    	/**
    	 * Generated serial version UID
    	 */
    	private static final long serialVersionUID = -8066394344204413879L;
    	private Panel all = new Panel();

    	/** parameters panel (whole left panel) */
    	JPanel paramsPanel = new JPanel();
    	/** clustering parameter panel (top left) */
    	JPanel clusteringPanel = new JPanel();
    	/** result options panel (bottom left) */
    	JPanel resultsPanel = new JPanel();

    	/** Panel with the channel options */
        private JPanel channelSelectionPanel = new JPanel();
        /** Panel with the clusterer selection */
        private JPanel clustererPanel = new JPanel();
        private JPanel executionPanel = new JPanel();
        /** Sample selection panel (for number of samples to use) */
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
                JSlider slider = (JSlider) samplePanel.getComponent(0);
                numSamples = ((image.getHeight()*image.getWidth())*image.getNSlices()) * slider.getValue() / 100;
                JTextArea textArea = (JTextArea) samplePanel.getComponent(1);
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
                exec.submit(new Runnable() {
                    public void run() {
                        ImagePlus result = clusteredImage.duplicate();
                        result.setCalibration(image.getCalibration());
                        result.setFileInfo(image.getFileInfo());
                        result.setTitle( inputImageShortTitle +"-clusters");
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
                        result.setTitle( inputImageShortTitle + "-clusterprobmap");
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
         * Action listener for channel selection, sets featuresCreated flag as false in order to force feature creation on next use.
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

            // === Channel panel ===
            // read number of available channels from ColorClustering class
            numChannels = ColorClustering.Channel.numChannels();
            channelSelectionPanel.setLayout(new GridLayout(3, numChannels/3, 6, 0));
            channelSelectionPanel.setBorder(BorderFactory.createTitledBorder("Channel"));
            channelSelectionPanel.setToolTipText("Choose channels to be used");
            // get list of channel names
            String[] channelList = ColorClustering.Channel.getAllLabels();
            // add them to the panel
            for(int i=0;i<numChannels;++i){
                JCheckBox tmp = new JCheckBox(channelList[i]);
                tmp.addActionListener(channelSelect);
                channelSelectionPanel.add(tmp,i);
            }

            // === Sample selection panel ===
            pixelSlider = new JSlider(1,100,50);
            samplePanel.setBorder(BorderFactory.createTitledBorder("Number of Samples"));
            samplePanel.setToolTipText("Select a percentage of pixels to be used when training the clusterer");
            samplePanel.add(pixelSlider,0);
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
            samplePanel.add(txtNumSamples,1);
            pixelSlider.addChangeListener(sampleChange);

            // === Clusterer panel ===
            GridBagConstraints clustererConstraints = new GridBagConstraints();
            clustererPanel.setLayout( new GridBagLayout() );
            clustererConstraints.insets = new Insets( 5, 5, 6, 6 );
            clustererConstraints.anchor = GridBagConstraints.CENTER;
            clustererConstraints.fill = GridBagConstraints.BOTH;
            clustererConstraints.gridwidth = 1;
            clustererConstraints.gridheight = 1;
            clustererConstraints.gridx = 0;
            clustererConstraints.gridy = 0;
            clustererConstraints.weightx = 0;
            clustererConstraints.weighty = 0;

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
            clustererConstraints.weightx = 1;
            clustererPanel.add(clustererEditorPanel, clustererConstraints);
            clustererConstraints.gridy++;
            clustererPanel.setBorder(BorderFactory.createTitledBorder("Clusterer"));
            clustererPanel.setToolTipText("Choose clusterer to be used");

            clusterizeButton = new JButton("Clusterize");
            JPanel clusterizeButtonPanel = new JPanel();
            clusterizeButton.setToolTipText("Clusterize the image!");
            clusterizeButtonPanel.add( clusterizeButton );
            clustererPanel.add(clusterizeButtonPanel, clustererConstraints);
            clustererConstraints.gridy++;

            clusterizeButton.addActionListener(clusterize);

            // === Execution panel ===
            GridBagConstraints executionConstraints = new GridBagConstraints();
            executionPanel.setLayout( new GridBagLayout() );
            executionConstraints.anchor = GridBagConstraints.CENTER;
            executionConstraints.fill = GridBagConstraints.BOTH;
            executionConstraints.gridwidth = 1;
            executionConstraints.gridheight = 1;
            executionConstraints.gridx = 0;
            executionConstraints.gridy = 0;
            executionConstraints.weightx = 0;
            executionConstraints.weighty = 0;

            createFile = new JButton("Create ARFF file");
            createFile.setToolTipText("Create a file");
            createFile.addActionListener(fileCreation);
            executionPanel.add(createFile, executionConstraints);
            executionConstraints.gridy++;

            toggleOverlay = new JButton("Toggle overlay");
            executionPanel.add(toggleOverlay, executionConstraints);
            executionConstraints.gridy++;
            toggleOverlay.setToolTipText("Toggle result image overlay!");
            toggleOverlay.addActionListener(overlay);
            toggleOverlay.setEnabled(false);

            JPanel opacityPanel = new JPanel();
            opacitySlider = new JSlider(0,100,50);
            opacityPanel.add(new Label("Overlay opacity:"));
            opacitySlider.setToolTipText("Select a percentage for the opacity");
            opacityPanel.add(opacitySlider);
            executionPanel.add( opacityPanel, executionConstraints );
            executionConstraints.gridy++;
            opacitySlider.addChangeListener(opacityChange);
            opacitySlider.setEnabled(false);

            createResult = new JButton("Show result");
            executionPanel.add(createResult, executionConstraints);
            executionConstraints.gridy++;
            createResult.addActionListener(resultCreation);
            createResult.setEnabled(false);

            createProbabilityMap = new JButton("Probability Map");
            executionPanel.add(createProbabilityMap, executionConstraints);
            executionConstraints.gridy++;
            createProbabilityMap.addActionListener(probMapCreator);
            createProbabilityMap.setEnabled(false);

            visualizeData = new JButton("Visualize data");
            executionPanel.add(visualizeData, executionConstraints);
            executionConstraints.gridy++;
            visualizeData.addActionListener(dataVisualizer);

            saveClusterer = new JButton("Save clusterer");
            executionPanel.add(saveClusterer, executionConstraints);
            executionConstraints.gridy++;
            saveClusterer.addActionListener(saveTheClusterer);
            saveClusterer.setEnabled(false);

            loadClusterer = new JButton("Load clusterer");
            executionPanel.add(loadClusterer, executionConstraints);
            executionConstraints.gridy++;
            loadClusterer.addActionListener(clusterLoader);

            // Clustering parameter panel (top left panel)
            GridBagLayout clusteringLayout = new GridBagLayout();
			GridBagConstraints clusteringConstraints = new GridBagConstraints();
			clusteringConstraints.insets = new Insets( 5, 5, 6, 6 );
			clusteringPanel.setLayout( clusteringLayout );
			clusteringConstraints.anchor = GridBagConstraints.NORTHWEST;
			clusteringConstraints.fill = GridBagConstraints.HORIZONTAL;
			clusteringConstraints.gridwidth = 1;
			clusteringConstraints.gridheight = 1;
			clusteringConstraints.gridx = 0;
			clusteringConstraints.gridy = 0;
			clusteringPanel.add( channelSelectionPanel, clusteringConstraints);
			clusteringConstraints.gridy++;
			clusteringPanel.add( samplePanel, clusteringConstraints);
			clusteringConstraints.gridy++;
			clusteringPanel.add( clustererPanel, clusteringConstraints);
			clusteringConstraints.gridy++;

			// Result options panel (bottom left panel)
			resultsPanel.setBorder(BorderFactory.createTitledBorder("Results"));
			resultsPanel.setToolTipText("Result options");
            GridBagLayout resultsLayout = new GridBagLayout();
			GridBagConstraints resultConstraints = new GridBagConstraints();
			resultConstraints.insets = new Insets( 5, 5, 6, 6 );
			resultsPanel.setLayout( resultsLayout );
			resultConstraints.anchor = GridBagConstraints.NORTHWEST;
			resultConstraints.fill = GridBagConstraints.HORIZONTAL;
			resultConstraints.gridwidth = 1;
			resultConstraints.gridheight = 1;
			resultConstraints.gridx = 0;
			resultConstraints.gridy = 0;
			resultsPanel.add( executionPanel, resultConstraints);
			resultConstraints.gridy++;

            // Whole left panel (parameters panel)
            GridBagLayout paramsLayout = new GridBagLayout();
			GridBagConstraints paramsConstraints = new GridBagConstraints();
			paramsConstraints.insets = new Insets( 5, 5, 6, 6 );
			paramsPanel.setLayout( paramsLayout );
			paramsConstraints.anchor = GridBagConstraints.NORTHWEST;
			paramsConstraints.fill = GridBagConstraints.HORIZONTAL;
			paramsConstraints.gridwidth = 1;
			paramsConstraints.gridheight = 1;
			paramsConstraints.gridx = 0;
			paramsConstraints.gridy = 0;
			paramsPanel.add( clusteringPanel, paramsConstraints);
			paramsConstraints.gridy++;
			paramsPanel.add( resultsPanel, paramsConstraints);
			paramsConstraints.gridy++;

			// main panel (including parameters panel and canvas)
			GridBagLayout layout = new GridBagLayout();
			GridBagConstraints allConstraints = new GridBagConstraints();
			all.setLayout(layout);

			// put parameter panel in place
			allConstraints.anchor = GridBagConstraints.NORTHWEST;
			allConstraints.fill = GridBagConstraints.BOTH;
			allConstraints.gridwidth = 1;
			allConstraints.gridheight = 1;
			allConstraints.gridx = 0;
			allConstraints.gridy = 0;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;

			all.add( paramsPanel, allConstraints );

			// put canvas in place
			allConstraints.gridx++;
			allConstraints.weightx = 1;
			allConstraints.weighty = 1;
			all.add( canvas, allConstraints );

			allConstraints.gridy++;
			allConstraints.weightx = 0;
			allConstraints.weighty = 0;

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
                                        image.updateAndDraw();

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
                                    image.updateAndDraw();
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
                                        image.updateAndDraw();
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
         * @return success or failure
         */
        boolean createFeatures(){
            boolean someChannelSelected = false;

            selectedChannels = new boolean[numChannels];
            numChannels = ColorClustering.Channel.numChannels();
            for (
                    int i = 0;
                    i < numChannels; ++i)

            {
                JCheckBox selected = (JCheckBox) channelSelectionPanel.getComponent(i);
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
         * Creates features and displays the data
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
                JCheckBox checkBox = (JCheckBox) win.channelSelectionPanel.getComponent(i);
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
            // Dialog closed by user
        	return;
        }else {
            IJ.log("Loading Weka properties");
            // store input image title
            inputImageTitle = image.getTitle();
            inputImageShortTitle = image.getShortTitle();
            // rename image so the plugin title is shown
            image.setTitle( "Color Clustering" );
            win = new CustomWindow(image);
        }

    }
}
