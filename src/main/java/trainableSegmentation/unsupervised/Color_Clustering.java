package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.StackWindow;
import ij.plugin.PlugIn;
import javafx.scene.control.CheckBox;
import jdk.nashorn.internal.runtime.arrays.NumericElements;
import trainableSegmentation.Weka_Segmentation;
import weka.clusterers.Clusterer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class Color_Clustering implements PlugIn{

    private int numClusters=3;//Default
    private int numSamples=3;//Default
    private ArrayList<ColorClustering.Channel> channels = new ArrayList<ColorClustering.Channel>();
    private ImagePlus ogImage;
    private CustomWindow win;
    private int numElements;//Number of elements on the GUI

    @Override
    public void run(String arg) {
        ogImage = IJ.openImage();
        win = new CustomWindow(ogImage);
        numElements=0;
        for(String element : ColorClustering.Channel.getAllLabels()){
            win.add(new JCheckBox(element),numElements);
            numElements++;
        }
        JButton clusterizer = new JButton("Clusterize!");
        win.add(clusterizer);
        TextField tfClusters = new TextField("Number of clusters");
        TextField tfSamples = new TextField("Number of samples");
        win.add(tfClusters,numElements);
        numElements++;
        win.add(tfSamples,numElements);
        numElements++;
        clusterizer.addActionListener(clusterize);
        IJ.log("Number of components"+win.getComponentCount());
        win.maximize();
        /*ImagePlus image = IJ.openImage();
        image.show();
        ColorClustering imageClustered = new ColorClustering(image,image.getHeight());
        int numClusters = (int) IJ.getNumber("Number of clusters!",2);
        PixelClustering pixelClusterer = new PixelClustering(imageClustered.getFeaturesInstances(),numClusters);
        Clusterer theClusterer = pixelClusterer.getClusterer();
        */
    }

    private class CustomWindow extends StackWindow
    {
        private JPanel mainPannel = new JPanel();

        public CustomWindow(ImagePlus imp) {
            super(imp);
        }
    }

    private ActionListener clusterize = new ActionListener() {
        public void actionPerformed(final ActionEvent e) {
            for(int i=0;i< numElements-3;++i){
                JCheckBox checkBox = (JCheckBox) win.getComponent(i);
                if(checkBox.isSelected()) {
                    channels.add(ColorClustering.Channel.fromLabel(checkBox.getText()));
                }
            }
            TextField clusterstxt = (TextField) win.getComponent(numElements-2);
            TextField samplestxt = (TextField) win.getComponent(numElements-1);
            numClusters = Integer.parseInt(clusterstxt.getText());
            numSamples = Integer.parseInt(samplestxt.getText());
            ColorClustering colorClustering = new ColorClustering(ogImage, numSamples, numClusters,channels);
            //colorClustering.createFile("sampled.arff",colorClustering.getFeaturesInstances());
            ImagePlus clusteredImage = colorClustering.createClusteredImage();
            clusteredImage.show();
        }
    };
}
