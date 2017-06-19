package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import weka.clusterers.Clusterer;

public class Color_Clustering implements PlugIn{

    private int numClusters;
    private int numSamples;

    @Override
    public void run(String arg) {
        ImagePlus image = IJ.openImage();
        image.show();
        numClusters =(int) IJ.getNumber("Number of clusters",2);
        numSamples = (int) IJ.getNumber("Number of samples",2);
        ColorClustering colorClustering = new ColorClustering(image,numSamples,numClusters);
        colorClustering.createFile("sampled.arff",colorClustering.getFeaturesInstances());
        ImagePlus clusteredImage = colorClustering.createClusteredImage();
        clusteredImage.show();
        /*ImagePlus image = IJ.openImage();
        image.show();
        ColorClustering imageClustered = new ColorClustering(image,image.getHeight());
        int numClusters = (int) IJ.getNumber("Number of clusters!",2);
        PixelClustering pixelClusterer = new PixelClustering(imageClustered.getFeaturesInstances(),numClusters);
        Clusterer theClusterer = pixelClusterer.getClusterer();
        */


    }
}
