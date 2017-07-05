package trainableSegmentation.unsupervised;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.ImageCanvas;
import ij.gui.StackWindow;
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
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.nio.channels.Channel;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Color_Clustering implements PlugIn{

    protected ImagePlus image;
    private boolean[] selectedChannels;
    private int numSamples;
    private int numChannels;
    private boolean file=false;
    private AbstractClusterer clusterer;
    ImagePlus displayImage = null;

    private CustomWindow win;

    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel channelSelection = new JPanel();
        private JPanel clusterizerSelection = new JPanel();
        private JPanel executor = new JPanel();
        private JPanel samplePanel = new JPanel();
        private GenericObjectEditor clustererEditor = new GenericObjectEditor();

        ChangeListener sampleChange = new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider slider = (JSlider) samplePanel.getComponent(1);
                numSamples = (image.getHeight()*image.getWidth()) * slider.getValue() / 100;
                JTextArea textArea = (JTextArea) samplePanel.getComponent(2);
                textArea.setText(Integer.toString(slider.getValue())+"% ("+numSamples+") " + "px");
            }
        };
        ActionListener clusterize = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                boolean someChannelSelected = false;
                Object c = ( Object ) clustererEditor.getValue();
                String options = "";
                String[] optionsArray = ((OptionHandler)c).getOptions();
                if ( c instanceof OptionHandler )
                {
                    options = Utils.joinOptions( optionsArray );
                }
                try{
                    clusterer = (AbstractClusterer) (c.getClass().newInstance());
                    clusterer.setOptions( optionsArray );
                }
                catch(Exception ex)
                {
                    IJ.log("Error when setting clusterer");
                }
                selectedChannels = new boolean[numChannels];
                numChannels = ColorClustering.Channel.numChannels();
                for(int i=0;i<numChannels;++i){
                    JCheckBox selected = (JCheckBox) channelSelection.getComponent(i);
                    selectedChannels[i] = selected.isSelected();
                    if(selected.isSelected()&&!someChannelSelected){
                        someChannelSelected=true;
                    }
                }
                if(someChannelSelected) {
                    IJ.log("Number of selected samples: "+numSamples);
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
                    ImagePlus clusteredImage = colorClustering.createClusteredImage(theFeatures);
                    clusteredImage.show();
                }else {
                    JOptionPane warning = new JOptionPane();
                    warning.showMessageDialog(all,"Choose at least a channel","Warning",JOptionPane.WARNING_MESSAGE);
                }

            }

        };

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

            }

            samplePanel.add(new Label("Select sample percentage:"));
            JSlider slider = new JSlider(1,100,50);
            samplePanel.add(slider,1);
            samplePanel.setBorder(BorderFactory.createTitledBorder("Number of Samples"));
            samplePanel.setToolTipText("Select a percentage of pixels to be used when training the clusterer");
            JTextArea txtNumSamples = new JTextArea("50% ("+Integer.toString(((image.getHeight()*image.getWidth()) * slider.getValue() / 100))+") px");
            numSamples=image.getHeight()*image.getWidth()* slider.getValue() / 100;
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
            clusterizerSelection.setToolTipText("Choose clusterere to be used");
            all.add(clusterizerSelection,allConstraints);
            allConstraints.gridy++;

            JButton execute = new JButton("Clusterize!");
            executor.add(execute);
            execute.setToolTipText("Clusterize the image!");
            all.add(executor,allConstraints);

            execute.addActionListener(clusterize);



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

        }
    }


    @Override
    public void run(String s) {

        image = WindowManager.getCurrentImage();
        if(image == null){
            image=IJ.openImage();
        }
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
