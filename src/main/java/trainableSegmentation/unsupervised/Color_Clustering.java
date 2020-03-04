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
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import java.awt.BorderLayout;
import java.awt.Component;
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
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
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
	/** executor service to manage plugin threads */
    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    /** input image */
    protected ImagePlus image=null;
    /** original image */
    protected ImagePlus originalImage=null;
    /** array of booleans indicating the selection of channels */
    private boolean[] selectedChannels;
    /** number of samples to use in the cluster building */
    private int numSamples;
    /** number of channels available */
    private int numChannels;
    /** clustering model to use */
    private AbstractClusterer clusterer=null;
    /** clusterer building thread */
    private Thread currentTask=null;
    /** result image with the output clusters */
    private ImagePlus clusteredImage=null;
    /** main plugin window */
    private CustomWindow win;
    /** color clustering backend */
    private ColorClustering colorClustering = null;
    /** flag indicating if the color features have been created */
    private boolean featuresCreated = false;
    /** array of color feature stacks (one per slice) */
    private FeatureStackArray featureStackArray = null;

    /** input image title */
    String inputImageTitle = null;
    /** input image short title */
    String inputImageShortTitle = null;
    /** flag to store if the clusterer finished */
    boolean finishedClustering = false;

    /**
     * Custom window based on JPanel structures
     */
    private class CustomWindow extends StackWindow
    {
    	/**
    	 * Generated serial version UID
    	 */
    	private static final long serialVersionUID = -8066394344204413879L;
    	/** panel containing the whole GUI (buttons on the left and canvas
    	 * on the right */
    	private Panel all = new Panel();

    	/** parameters panel (whole left panel) */
    	JPanel paramsPanel = new JPanel();
    	/** clustering parameter panel (top left) */
    	JPanel clusteringPanel = new JPanel();
    	/** result options panel (bottom left) */
    	JPanel resultsPanel = new JPanel();

    	/** panel with the channel options */
        private JPanel channelSelectionPanel = new JPanel();
        /** panel with the clusterer selection */
        private JPanel clustererPanel = new JPanel();
        /** sample selection panel (for number of samples to use) */
        private JPanel samplePanel = new JPanel();
        /** array of channel checkboxes */
        JCheckBox[] channelCheckbox = null;
        /** Weka clusterer edition panel */
        private GenericObjectEditor clustererEditor = new GenericObjectEditor();
        /** button to run (and stop) clusterer building */
        private JButton runClusterButton = null;
        /** button to create ARFF file from current settings */
        private JButton createFile = null;
        /** button to display current result segmentation */
        private JButton createResult = null;
        /** button to create probability map image */
        private JButton createProbabilityMap = null;
        /** button to visualize data in Weka visualization window */
        private JButton visualizeData = null;
        /** button to save clustere to file (.model) */
        private JButton saveClusterer = null;
        /** button to load clusterer from file (.model) */
        private JButton loadClusterer = null;
        /** flag to warn about large number of samples */
        private boolean warned=false;
        /** slider to select the number of samples to use */
        private JSlider pixelSlider;
        /** slider to change the overlay opacity */
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
         * Action listener for Run button
         */
        ActionListener runButtonListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String command = e.getActionCommand();
                exec.submit(new Runnable() {
                    public void run() {
                        if(e.getSource()==runClusterButton) {
                            runClusterOrStop(command);
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
                        SaveDialog sd = new SaveDialog(
                     		   "Save features as...", "features", ".arff" );
                        if( null != sd.getDirectory() )
                        {
                        	if( !featuresCreated && !createFeatures() ) {
                            	IJ.log( "Error while creating features!" );
                            	return;
                            }
                        	String path = sd.getDirectory() + sd.getFileName();
                        	IJ.log( "Saving features to ARFF file...");
                        	if( colorClustering.createFile( path,
                        			colorClustering.getFeaturesInstances()) )
                        		IJ.log( "Saved features as " + path );
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
                        if( !featuresCreated )
                    	{
                        	IJ.log( "Creating features..." );
                        	createFeatures();
                    	}
                        if( clusterer == null )
                            updateClusterer();

                        if( !finishedClustering )
                        	buildClusterer();
                        // Generate probability map image
                        IJ.log( "Creating probability map image..." );
                        ImagePlus result = colorClustering.createProbabilityMaps(featureStackArray);
                        result.setCalibration(image.getCalibration());
                        result.setFileInfo(image.getFileInfo());
                        result.setTitle( inputImageShortTitle + "-clusterprobmap");
                        result.show();
                        IJ.log( "Done" );
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
                        visualizeData();
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
            	exec.submit(new Runnable() {
            		public void run() {
            			// Disable GUI components until loading is finished
                    	win.enableComponents( false );
            			loadClusterer();
            			win.updateComponentEnabling();
            		}
            	});
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
            channelCheckbox = new JCheckBox[ numChannels ];
            for(int i=0;i<numChannels;++i){
            	channelCheckbox[ i ] = new JCheckBox(channelList[i]);
            	channelCheckbox[ i ].addActionListener(channelSelect);
                channelSelectionPanel.add( channelCheckbox[ i ], i );
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

            runClusterButton = new JButton("Run");
            JPanel runClusterButtonPanel = new JPanel();
            runClusterButton.setToolTipText("Cluster the image!");
            runClusterButtonPanel.add( runClusterButton );
            clustererPanel.add(runClusterButtonPanel, clustererConstraints);
            clustererConstraints.gridy++;

            runClusterButton.addActionListener(runButtonListener);

            // === Overlay panel (toggle button and slider) ===
            GridBagConstraints overlayConstraints = new GridBagConstraints();
            JPanel overlayPanel = new JPanel();
            overlayPanel.setLayout( new GridBagLayout() );
            overlayConstraints.anchor = GridBagConstraints.CENTER;
            overlayConstraints.fill = GridBagConstraints.BOTH;
            overlayConstraints.gridwidth = 1;
            overlayConstraints.gridheight = 1;
            overlayConstraints.gridx = 0;
            overlayConstraints.gridy = 0;
            overlayConstraints.weightx = 0;
            overlayConstraints.weighty = 0;
            overlayConstraints.insets = new Insets( 5, 5, 6, 6 );

            // Opacity slider
            JPanel opacityPanel = new JPanel();
            opacitySlider = new JSlider(0,100,50);
            opacityPanel.add(new Label("Opacity:"));
            opacitySlider.setToolTipText("Select a percentage for the opacity");
            opacityPanel.add(opacitySlider);
            overlayPanel.add( opacityPanel, overlayConstraints );
            overlayConstraints.gridy++;
            opacitySlider.addChangeListener(opacityChange);
            opacitySlider.setEnabled(false);

            // Input/Output options panel
            JPanel ioPanel = new JPanel();
            GridBagConstraints ioConstraints = new GridBagConstraints();
            ioPanel.setLayout( new GridBagLayout() );
            ioConstraints.anchor = GridBagConstraints.CENTER;
            ioConstraints.fill = GridBagConstraints.BOTH;
            ioConstraints.gridwidth = 1;
            ioConstraints.gridheight = 1;
            ioConstraints.gridx = 0;
            ioConstraints.gridy = 0;
            ioConstraints.weightx = 0;
            ioConstraints.weighty = 0;
            ioConstraints.insets = new Insets( 5, 5, 6, 6 );

            createFile = new JButton("Create ARFF file");
            createFile.setToolTipText("Create a file");
            createFile.addActionListener(fileCreation);
            ioPanel.add(createFile, ioConstraints);
            ioConstraints.gridy++;

            createResult = new JButton("Show result");
            ioPanel.add(createResult, ioConstraints);
            ioConstraints.gridy++;
            createResult.addActionListener(resultCreation);
            createResult.setEnabled(false);

            createProbabilityMap = new JButton("Probability Map");
            ioPanel.add(createProbabilityMap, ioConstraints);
            ioConstraints.gridy++;
            createProbabilityMap.addActionListener(probMapCreator);
            createProbabilityMap.setEnabled(false);

            visualizeData = new JButton("Visualize data");
            ioPanel.add(visualizeData, ioConstraints);
            ioConstraints.gridy++;
            visualizeData.addActionListener(dataVisualizer);

            saveClusterer = new JButton("Save clusterer");
            ioPanel.add(saveClusterer, ioConstraints);
            ioConstraints.gridy++;
            saveClusterer.addActionListener(saveTheClusterer);
            saveClusterer.setEnabled(false);

            loadClusterer = new JButton("Load clusterer");
            ioPanel.add(loadClusterer, ioConstraints);
            ioConstraints.gridy++;
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
			resultsPanel.add( overlayPanel, resultConstraints);
			resultConstraints.gridy++;
			resultsPanel.add( ioPanel, resultConstraints);
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
            if( null != super.sliceSelector )
            {
                sliceSelector.setValue( image.getCurrentSlice() );
                image.setSlice( image.getCurrentSlice() );

                all.add( super.sliceSelector, allConstraints );
                if( null != super.zSelector ) {
                    all.add(super.zSelector, allConstraints);
                }
                if( null != super.tSelector ){
                    all.add( super.tSelector, allConstraints );
                }
                if( null != super.cSelector ){
                    all.add( super.cSelector, allConstraints );
                }
            }
            allConstraints.gridy--;

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

            // Add window listener to close things properly
            addWindowListener( new WindowAdapter() {
        	public void windowClosing(WindowEvent e) {
        	    super.windowClosing( e );
        	    // cleanup
        	    if( null != originalImage )
        	    {
        		// display training image
        		if( null == originalImage.getWindow() )
        		    originalImage.show();
        		originalImage.getWindow().setVisible( true );
        	    }
        	    // Stop any thread from the clusterer
        	    if( null != currentTask )
        	    {
        		currentTask.interrupt();
        		currentTask.stop();
        	    }
        	    exec.shutdownNow();
        	}
            });
            
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
                                	updateResultOverlay();
                                	image.updateAndDraw();
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
                            	updateResultOverlay();
                            	image.updateAndDraw();
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
                                	updateResultOverlay();
                                	image.updateAndDraw();
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

        }// end CustomWindow constructor
        /**
         * Enable/disable all GUI components
         * @param flag boolean flag to enable or disable all GUI components
         */
        void enableComponents( boolean flag )
        {
        	this.channelSelectionPanel.setEnabled( flag );
        	for( int i=0; i<channelCheckbox.length; i++ )
        		channelCheckbox[ i ].setEnabled( flag );
        	this.pixelSlider.setEnabled( flag );
        	this.samplePanel.setEnabled( flag );
        	this.clustererPanel.setEnabled( flag );
        	this.clustererEditor.setEnabled( flag );
        	for( Component c : this.clustererEditor.getCustomPanel().getComponents() )
        		c.setEnabled( flag );
        	this.runClusterButton.setEnabled( flag );
        	this.opacitySlider.setEnabled( flag );
        	this.createFile.setEnabled( flag );
        	this.createResult.setEnabled( flag );
        	this.createProbabilityMap.setEnabled( flag );
        	this.visualizeData.setEnabled( flag );
        	this.saveClusterer.setEnabled( flag );
        	this.loadClusterer.setEnabled( flag );
        }
        /**
         * Update component enabling based on plugin status
         */
        void updateComponentEnabling()
        {
        	this.channelSelectionPanel.setEnabled( true );
        	for( int i=0; i<channelCheckbox.length; i++ )
        		channelCheckbox[ i ].setEnabled( true );
        	this.pixelSlider.setEnabled( true );
        	this.samplePanel.setEnabled( true );
        	this.clustererPanel.setEnabled( true );
        	this.clustererEditor.setEnabled( true );
        	this.runClusterButton.setEnabled( true );
        	this.opacitySlider.setEnabled( true );
        	this.createFile.setEnabled( true );
        	this.createResult.setEnabled( finishedClustering );
        	this.createProbabilityMap.setEnabled( finishedClustering );
        	this.visualizeData.setEnabled( true );
        	this.saveClusterer.setEnabled( finishedClustering );
        	this.loadClusterer.setEnabled( true );
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
                // Update channels to use and number of samples
                colorClustering.setNumSamples( numSamples );
                colorClustering.setChannels( channels );
                IJ.log("Creating features...");
                colorClustering.createFeatures();
                featureStackArray = colorClustering.getFeatureStackArray();
                featuresCreated = true;
                return true;
            } else {
                IJ.error("Warning!","Choose at least a channel");
                runClusterButton.setText("Run");
                return false;
            }
        }

        /**
         * Creates features and displays the data
         */
        void visualizeData(){
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
         * Cluster the selected image if the command is "Run", else it stops the clustering.
         * WARNING: Weka does not support usage of interrupt flag and instead uses deprecated stop function.
         * @param command
         */
		void runClusterOrStop(String command){
            if(command.equals("Run")){
            	// disable GUI components when running clusterer
            	enableComponents( false );
            	runClusterButton.setEnabled( true );
                runClusterButton.setText("STOP");
                final Thread oldTask = currentTask;
                Thread newTask = new Thread() {

                    public void run() {

                        if (null != oldTask)
                        {
                            try {
                                // Waiting for old task to finish
                                oldTask.join();
                            }
                            catch (InterruptedException ie)	{ IJ.log("interrupted"); }
                        }
                        clusterer = colorClustering.getTheClusterer();
                        updateClusterer();
                        // Create color features when needed
                        if( !featuresCreated )
                        {
                        	if( !createFeatures() )
                        	{
                        		finishedClustering = false;
                            	updateComponentEnabling();
                            	return;
                        	}
                        }
                        // Build clusterer
                        buildClusterer();
                        IJ.log("Creating clustered image...");
                        clusteredImage = colorClustering.createClusteredImage(featureStackArray);
                        updateResultOverlay();
                        runClusterButton.setText("Run");
                        enableComponents( true );
                        finishedClustering = true;
                        IJ.log( "Done" );
                    }
                };
                currentTask = newTask;
                newTask.start();
            }else if(command.equals("STOP")){
                IJ.log("Clustering stopped by user.");
                runClusterButton.setText("Run");
                // enable GUI components
            	enableComponents( true );
                if(null != currentTask) {
                    currentTask.interrupt();//Should use interrupt but weka does not support interrupt handling.
                    currentTask.stop();//Interrupt is being used
                }else{
                    IJ.log("Error: Interrupting failed because thread was null!");
                }
            }
        }

        /**
         * Updates clusterer based on selected options on GUI
         */
        public void updateClusterer()
        {
        	Object c = (Object) clustererEditor.getValue();
        	String[] optionsArray = ((OptionHandler) c).getOptions();
            boolean update = false;
            
            if(clusterer!=null) {
            	String[] prevOptions = clusterer.getOptions();
            	// If different length of options, assume different clusterer
            	if (optionsArray.length != prevOptions.length)
            		update = true;
            	else {
            		for (int i = 0; i < optionsArray.length; ++i) {
            			if (!prevOptions[i].contentEquals(optionsArray[i])) {
            				update = true;
            				break;
            			}
            		}
            	}
            }
            else
            	update = true;

            if( update )
            {
            	try {
            		clusterer = (AbstractClusterer) (c.getClass().newInstance());
            		clusterer.setOptions(optionsArray);
            	}
            	catch ( Exception ex )
            	{
            		ex.printStackTrace();
            		runClusterButton.setText("Run");
            		IJ.log("Error when setting clusterer.");
            	}
            }
        }
        /**
         * Build current clusterer
         */
		public void buildClusterer() {
			IJ.log("Building clusterer...");
			long startTime = System.currentTimeMillis();
			AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
			long endTime = System.currentTimeMillis();
			colorClustering.setTheClusterer(theClusterer);
			IJ.log(theClusterer.toString());
			IJ.log( "Clusterer building took " + (endTime - startTime) + " ms." );
		}
    } // end class CustomWindow

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
        }
        else
        {        	
            // Update current clusterer        	
            clusterer = colorClustering.getTheClusterer();
            // Match selected features in GUI with those used to build
            // the clusterer
            Instances featuresInstances = colorClustering.getFeaturesInstances();
            boolean[] enabledChannels = new boolean[numChannels];
            for(int i=0;i<numChannels;++i){
                enabledChannels[i]=false;
            }
            for(int i=0;i<featuresInstances.numAttributes();++i){
                String name = featuresInstances.attribute(i).name();
                for(int j = 0; j < numChannels; ++j )
                {
                	if( win.channelCheckbox[ j ].getText().equals( name ) )
                	{
                		enabledChannels[ j ] = true;
                		break;
                	}
                }
            }
            // Update checkbox with enabled channels
            for(int i=0;i<numChannels;++i)
            {
            	// Set flag to create features if any channel is different
            	if( win.channelCheckbox[ i ].isSelected() !=  enabledChannels[i] )
            		featuresCreated = false;
                win.channelCheckbox[ i ].setSelected(enabledChannels[i]);
            }
            // Update clusterer edition panel
            win.clustererEditor.setValue( clusterer );
            IJ.log("Loaded clusterer:" + clusterer);
            // New clusterer is already built
            finishedClustering = true;
            // Update plugin overlay
            if( !featuresCreated )
            {
            	if( !win.createFeatures() )
            	{
            		finishedClustering = false;            		
                	return;
            	}
            }
            IJ.log("Creating new clustered image...");
            clusteredImage = colorClustering.createClusteredImage(featureStackArray);
            win.updateResultOverlay();
            IJ.log("Done");
        }
    }


    /**
     * Run function for plug-in
     * @param s
     */
    @Override
    public void run(String s) {

    	if( null == WindowManager.getCurrentImage() ){
    		originalImage = IJ.openImage();

    		if( originalImage == null )
    			// Dialog closed by user
    			return;
    	}
    	else
    	{
    		originalImage = WindowManager.getCurrentImage();

    		// hide input image (to avoid accidental closing)
    		originalImage.getWindow().setVisible( false );
    	}
    	// store original input image title
    	inputImageTitle = originalImage.getTitle();
    	inputImageShortTitle = originalImage.getShortTitle();
    	// create a copy of the original image to be displayed
    	image = originalImage.duplicate();
    	image.setSlice( originalImage.getSlice() );
    	// rename image so the plugin title is shown
    	image.setTitle( "Color Clustering" );
    	IJ.log( "Loading all available Weka clustering methods..." );
    	IJ.showStatus( "Loading all available Weka clustering methods..." );

    	// Build GUI
    	SwingUtilities.invokeLater(
    			new Runnable() {
    				public void run() {
    					win = new CustomWindow( image );
    					win.pack();
    				}
    			});

    }
}
