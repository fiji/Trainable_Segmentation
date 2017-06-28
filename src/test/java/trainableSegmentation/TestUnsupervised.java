package trainableSegmentation;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.NewImage;
import ij.process.ImageProcessor;
import trainableSegmentation.unsupervised.ColorClustering;
import trainableSegmentation.unsupervised.PixelClustering;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

public class TestUnsupervised {

    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        image.show();
        ArrayList<ColorClustering.Channel> channels = new ArrayList<ColorClustering.Channel>();
        channels.add(ColorClustering.Channel.fromLabel("Lightness"));
        channels.add(ColorClustering.Channel.fromLabel("a"));
        channels.add(ColorClustering.Channel.fromLabel("b"));
        channels.add(ColorClustering.Channel.fromLabel("Red"));
        channels.add(ColorClustering.Channel.fromLabel("Green"));
        channels.add(ColorClustering.Channel.fromLabel("Blue"));
        channels.add(ColorClustering.Channel.fromLabel("Hue"));
        channels.add(ColorClustering.Channel.fromLabel("Brightness"));
        channels.add(ColorClustering.Channel.fromLabel("Saturation"));
        ColorClustering colorClustering = new ColorClustering(image,30, channels);
        SimpleKMeans kmeans = new SimpleKMeans();
        try {
            kmeans.setNumClusters(3);
        } catch (Exception e) {
            e.printStackTrace();
        }
        AbstractClusterer theClusterer = colorClustering.createClusterer(kmeans);
        colorClustering.setTheClusterer(theClusterer);
        colorClustering.createFile("test.arff",colorClustering.getFeaturesInstances());
        FeatureStackArray theFeatures = colorClustering.createFSArray(image);
        ImagePlus clusteredImage = colorClustering.createClusteredImage(theFeatures);
        clusteredImage.show();
    }



}

