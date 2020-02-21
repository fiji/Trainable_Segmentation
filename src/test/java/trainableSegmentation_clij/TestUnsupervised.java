package trainableSegmentation_clij;
import ij.IJ;
import ij.ImagePlus;
import trainableSegmentation_clij.unsupervised.ColorClustering;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Canopy;

import java.util.ArrayList;

/**
 * Test class for unsupervised learning in color images
 *
 */
public class TestUnsupervised {

    /**
     * Main method of test class
     *
     * @param args main arguments (not used)
     */
    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        image.show();
        ArrayList<trainableSegmentation_clij.unsupervised.ColorClustering.Channel> channels = new ArrayList<trainableSegmentation_clij.unsupervised.ColorClustering.Channel>();
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Lightness"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("a"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("b"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Red"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Green"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Blue"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Hue"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Brightness"));
        channels.add(trainableSegmentation_clij.unsupervised.ColorClustering.Channel.fromLabel("Saturation"));
        trainableSegmentation_clij.unsupervised.ColorClustering colorClustering = new ColorClustering(image,14, channels);
        Canopy clusterer = new Canopy();
        try {
            clusterer.setNumClusters(2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
        colorClustering.setTheClusterer(theClusterer);
        //colorClustering.createFile(colorClustering.getFeaturesInstances());
        FeatureStackArray theFeatures = colorClustering.createFSArray(image);
        ImagePlus probMap = colorClustering.createProbabilityMaps(colorClustering.getFeatureStackArray());
        probMap.show();
        ImagePlus clusteredImage = colorClustering.createClusteredImage(theFeatures);
        clusteredImage.show();
    }



}

