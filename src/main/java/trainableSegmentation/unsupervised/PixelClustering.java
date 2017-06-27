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

    public enum SelectedClusterer{
        Canopy("Canopy"),
        Cobweb("Cobweb"),
        EM("EM"),
        FarthestFirst("FarthestFirst"),
        FilteredClusterer("FilteredClusterer"),
        HierarchicalClusterer("HierarchicalClusterer"),
        MakeDensityBasedClusterer("MakeDensityBasedClusterer"),
        SimpleKMeans("SimpleKMeans");

        private final String label;

        private SelectedClusterer(String label){ this.label = label;}
        public String toString() {return this.label;}
        public static int numClusterers(){
            int number=0;
            for(String item : getAllClusterers()){
                number++;
            }
            return number;
        }
        public static String[] getAllClusterers(){
            int n = SelectedClusterer.values().length;
            String[] result = new String[n];
            int i=0;
            for(SelectedClusterer ch : SelectedClusterer.values()){
                result[i++] = ch.label;
            }
            return result;
        }
        public static SelectedClusterer fromLabel(String clLabel){
            if(clLabel != null){
                clLabel = clLabel.toLowerCase();
                for(SelectedClusterer ch : SelectedClusterer.values()){
                    String cmp = ch.label.toLowerCase();
                    if(cmp.equals(clLabel)){
                        return ch;
                    }
                }
                throw new IllegalArgumentException("Unable to parse Clusterer with label: " + clLabel);
            }
            return null;
        }
    }

    private AbstractClusterer clusterer;
    private Instances featuresInstances;
    private int numClusters;//Number of clusters
    private SelectedClusterer selectedClusterer;



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
    public PixelClustering(Instances featuresInstances,int numClusters, String clusterer){
        this.setNumClusters(numClusters);
        this.setFeaturesInstances(featuresInstances);
        this.setSelectedClusterer(SelectedClusterer.fromLabel(clusterer));
    }

    /**
     * Builds clusterer based on featuresInstances and number of clusters
     */
    public void buildClusterer(){
        Random rand = new Random();
        switch (this.getSelectedClusterer()){
            case Canopy:
                Canopy canopy = new Canopy();
                canopy.setSeed(rand.nextInt());
                try {
                    canopy.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    canopy.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(canopy.toString());
                this.clusterer = canopy;
                break;
            case Cobweb:
                Cobweb cobweb = new Cobweb();
                cobweb.setSeed(rand.nextInt());
                try {
                    cobweb.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(cobweb.toString());
                this.clusterer = cobweb;
                break;
            case EM:
                EM em = new EM();
                em.setSeed(rand.nextInt());
                try {
                    em.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    em.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(em.toString());
                this.clusterer = em;
                break;
            case FarthestFirst:
                FarthestFirst farthestFirst = new FarthestFirst();
                farthestFirst.setSeed(rand.nextInt());
                try {
                    farthestFirst.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    farthestFirst.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(farthestFirst.toString());
                this.clusterer = farthestFirst;
                break;
            case FilteredClusterer:
                FilteredClusterer filteredClusterer = new FilteredClusterer();
                try {
                    filteredClusterer.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(filteredClusterer.toString());
                this.clusterer = filteredClusterer;
                break;
            case HierarchicalClusterer:
                HierarchicalClusterer hierarchicalClusterer = new HierarchicalClusterer();
                try {
                    hierarchicalClusterer.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    hierarchicalClusterer.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(hierarchicalClusterer.toString());
                this.clusterer = hierarchicalClusterer;
                break;
            case MakeDensityBasedClusterer:
                MakeDensityBasedClusterer makeDensityBasedClusterer = new MakeDensityBasedClusterer();
                try {
                    makeDensityBasedClusterer.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    makeDensityBasedClusterer.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(makeDensityBasedClusterer.toString());
                this.clusterer = makeDensityBasedClusterer;
                break;
            case SimpleKMeans:
                SimpleKMeans kMeans = new SimpleKMeans();
                kMeans.setSeed(rand.nextInt());
                try {
                    kMeans.setNumClusters(numClusters);
                } catch (Exception e) {
                    IJ.log("Error when setting number of clusters");
                }
                try {
                    kMeans.buildClusterer(featuresInstances);
                } catch (Exception e) {
                    IJ.log("Error when building clusterer");
                }
                IJ.log("Clusterer built succesfully!");
                IJ.log(kMeans.toString());
                this.clusterer = kMeans;
                break;
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

    public SelectedClusterer getSelectedClusterer() {
        return selectedClusterer;
    }

    public void setSelectedClusterer(SelectedClusterer selectedClusterer) {
        this.selectedClusterer = selectedClusterer;
    }

}
