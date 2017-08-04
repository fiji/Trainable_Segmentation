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
import org.apache.commons.math3.analysis.function.Abs;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.WekaSegmentation;
import weka.clusterers.*;
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


    private Instances featuresInstances;
    private AbstractClusterer selectedClusterer;



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
    public PixelClustering(Instances featuresInstances, AbstractClusterer clusterer){
        this.setFeaturesInstances(featuresInstances);
        this.setSelectedClusterer(clusterer);
    }

    /**
     * Builds clusterer based on featuresInstances and number of clusters
     */
    public void buildClusterer(){
        IJ.log("Building clusterer");
        try {
            selectedClusterer.buildClusterer(featuresInstances);
        } catch (InterruptedException ie) {
            IJ.log("Clusterer building was interrupted");
        } catch (Exception e) {
            IJ.log(e.getMessage());
            e.printStackTrace();
        }

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
     * Get selected clusterer
     * @return
     */
    public AbstractClusterer getSelectedClusterer() {
        return selectedClusterer;
    }

    /**
     * Set selected clusterer
     * @param selectedClusterer
     */
    public void setSelectedClusterer(AbstractClusterer selectedClusterer) {
        this.selectedClusterer = selectedClusterer;
    }

}
