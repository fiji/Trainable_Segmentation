package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Line;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.process.ByteProcessor;
import ij.process.ColorSpaceConverter;
import ij.process.FloatPolygon;
import ij.process.ImageProcessor;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.WekaSegmentation;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

public class PixelClustering {

    private AbstractClusterer clusterer;
    private Instances featuresInstances;
    private int numClusters;//Number of clusters


    /**
     * Empty constructor
     */
    public PixelClustering(){

    }

    /**
     * Creates clusterer based on number of clusters and featuresInstances
     * @param featuresInstances
     * @param numClusters Because it uses K-Means
     */
    public PixelClustering(Instances featuresInstances,int numClusters){
        this.setNumClusters(numClusters);
        this.setFeaturesInstances(featuresInstances);
    }

    /**
     * Builds clusterer based on featuresInstances and number of clusters
     */
    public void buildClusterer(){
        SimpleKMeans theClusterer = new SimpleKMeans();
        Random rand = new Random();
        theClusterer.setSeed(rand.nextInt());
        try {
            theClusterer.setNumClusters(numClusters);
        } catch (Exception e) {
            IJ.log("Error when setting number of clusters");
        }
        try {
            theClusterer.buildClusterer(featuresInstances);
        } catch (Exception e) {
            IJ.log("Error when building clusterer");
        }
        IJ.log("Clusterer built succesfully!");
        IJ.log(theClusterer.toString());
        this.clusterer = theClusterer;
    }


    //Getters and setters

    /**
     * Set features instances to be used
     * @param featuresInstances
     */
    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }

    /**
     * Get features instances that are being used
     * @return
     */
    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    /**
     * Set clusterer to be used
     * @param clusterer
     */
    public void setClusterer(AbstractClusterer clusterer) {
        this.clusterer = clusterer;
    }

    /**
     * Get clusterer that has been created
     * @return
     */
    public AbstractClusterer getClusterer() {
        return clusterer;
    }

    /**
     * Set number of clusters
     * @param numClusters
     */
    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    /**
     * Get number of clusters
     * @return
     */
    public int getNumClusters() {
        return numClusters;
    }
}
