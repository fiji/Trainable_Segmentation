package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.StackWindow;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import javafx.scene.control.CheckBox;
import jdk.nashorn.internal.runtime.arrays.NumericElements;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.Weka_Segmentation;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Color_Clustering implements PlugIn{

    protected ImagePlus image;
    private boolean[] selectedChannels;
    private int numClusters;
    private int numSamples;
    private int numChannels;

    @Override
    public void run(String s) {

        image = WindowManager.getCurrentImage();
        if(image == null){
            image=IJ.openImage();
        }
        image.show();
        if(showDialog()){
            process();
        }
    }

    private boolean showDialog() {
        boolean someSelected = false;
        GenericDialog gd = new GenericDialog("Clusterize");

        gd.addNumericField("Number of clusters", 3,0);
        gd.addNumericField("Number of samples",30,0);
        numChannels = ColorClustering.Channel.numChannels();
        selectedChannels = new boolean[numChannels];
        for(int i=0;i<numChannels;++i){
            selectedChannels[i]=false;
        }
        gd.addCheckboxGroup(numChannels,1,ColorClustering.Channel.getAllLabels(),selectedChannels);
        gd.showDialog();
        if(gd.wasCanceled()){
            return false;
        }
        numClusters = (int) gd.getNextNumber();
        numSamples = (int) gd.getNextNumber();
        Vector<Checkbox> checkboxes = gd.getCheckboxes();
        for(int i=0;i<numChannels;++i){
            selectedChannels[i] = checkboxes.get(i).getState();
            if(checkboxes.get(i).getState()){
                someSelected=true;
            }
        }
        if(someSelected){
            IJ.log("Finished getting elements");
            return true;
        }else{
            IJ.log("Select at least a channel");
            return false;
        }
    }
    public void process(){
        IJ.log("Starting Processing");
        ArrayList<ColorClustering.Channel> channels = new ArrayList<ColorClustering.Channel>();
        for(int i=0;i<numChannels;++i){
            if(selectedChannels[i]){
                ColorClustering.Channel channel = ColorClustering.Channel.fromLabel(ColorClustering.Channel.getAllLabels()[i]);
                channels.add(channel);
            }
        }
        ColorClustering colorClustering = new ColorClustering(image,numSamples,channels);
        AbstractClusterer theClusterer = colorClustering.createClusterer(numClusters);
        colorClustering.setTheClusterer(theClusterer);
        FeatureStackArray theFeatures = colorClustering.createFSArray(image);
        ImagePlus clusteredImage = colorClustering.createClusteredImage(theFeatures);
        clusteredImage.show();
    }

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
