/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2021 Fiji developers.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */
package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.ByteProcessor;
import ij.process.ColorSpaceConverter;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.ReusableDenseInstance;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.awt.Point;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * This class contains all the library methods to perform
 * color clustering based on a few color space transformations
 * and the Weka clusterers.
 * @author Josu Salinas and Ignacio Arganda-Carreras
 */
public class ColorClustering {


    /**
     * Names of all channels (features) that can be used.
     * */
    public enum Channel{
        /** Red channel from RGB color space */
        Red("Red"),
        /** Green channel from RGB color space */
        Green("Green"),
        /** Blue channel from RGB color space */
        Blue("Blue"),
        /** L channel from Lab color space */
        Lightness("Lightness"),
        /** a channel from Lab color space */
        a("a"),
        /** b channel from Lab color space */
        b("b"),
        /** Hue channel from HSB color space */
        Hue("Hue"),
        /** Saturation channel from HSB color space */
        Saturation("Saturation"),
        /** Brightness channel from HSB color space */
        Brightness("Brightness");

    	/** channel label */
        private final String label;

        /**
         * Create channel with a label.
         * @param label
         */
        private Channel(String label){
            this.label = label;
        }

        /**
         * Turn channel to string.
         * @return channel label as a string
         */
        public String toString(){
            return this.label;
        }

        /**
         * Returns total number of channels, static method.
         * @return number of channels available
         */
        public static int numChannels(){
            return getAllLabels().length;
        }

        /**
         * Get all labels in a String[] structure.
         * @return array of channel names
         */
        public static String[] getAllLabels(){
            int n = Channel.values().length;
            String[] result = new String[n];
            int i=0;
            for(Channel ch : Channel.values()){
                result[i++] = ch.label;
            }
            return result;
        }

        /**
         * Get channel from label (label is a String).
         * @param chLabel
         * @return channel object specified by input label
         */
        public static Channel fromLabel(String chLabel){
            if(chLabel != null){
                chLabel = chLabel.toLowerCase();
                for(Channel ch : Channel.values()){
                    String cmp = ch.label.toLowerCase();
                    if(cmp.equals(chLabel)){
                        return ch;
                    }
                }
                throw new IllegalArgumentException("Unable to parse Channel with label: " + chLabel);
            }
            return null;
        }

    };
    /** list of channels to use as features */
    private ArrayList<Channel> channels = new ArrayList<Channel>();
    /** instances to build the clusterer */
    private Instances featuresInstances;
    /** color image to clusterize */
    private ImagePlus image;
    /** array of feature stacks used to store the image features of each 2D slice */
    private FeatureStackArray featureStackArray;
    /** number of samples to use to build the clusterer */
    private int numSamples;
    /** clustering Weka model used to cluster color image */
    private AbstractClusterer theClusterer;


    /**
     * Constructor using only image, features and clusterer will be set as null
     * @param image input color image to clusterize
     */
    public ColorClustering(ImagePlus image){
        featuresInstances=null;
        this.image=image;
        featureStackArray=null;
        numSamples=image.getHeight()*image.getWidth()*image.getNSlices()/2;
        theClusterer=null;
    }


    /**
     * Creates color clustering object based on image, number of samples and selected channels. Creates the features.
     * @param image input color image
     * @param numSamples number of samples to use
     * @param selectedChannels list of color space channels (features) to use
     */
    public ColorClustering(ImagePlus image, int numSamples, ArrayList<Channel> selectedChannels){
        for(Channel element: selectedChannels){
            this.channels.add(element);
        }
        this.setImage(image);
        this.setNumSamples(numSamples);
        featureStackArray = new FeatureStackArray(image.getStackSize());
        this.createFeatures();
    }

    /**
     * Build clusterer using the current features and the provided number of clusters.
     * @param selectedClusterer clustering Weka model to build
     * @return built clustering Weka model
     */
    public AbstractClusterer createClusterer(AbstractClusterer selectedClusterer){
        PixelClustering pixelClustering = new PixelClustering(this.getFeaturesInstances(),selectedClusterer);
        pixelClustering.buildClusterer();
        AbstractClusterer clusterer = pixelClustering.getSelectedClusterer();
        return clusterer;
    }


    /**
     * Generate the features based on the selected color space channels.
     */
    public void createFeatures(){
        int numSlices = image.getNSlices();
        int samplesPerSlice = numSamples/numSlices;
        // Initialize feature stack array
        featureStackArray = new FeatureStackArray(image.getStackSize());

        for(int slice = 1; slice <= image.getStackSize(); ++slice){
            boolean labactive=false,rgbactive=false,hsbactive=false;
            ImageConverter ic,ic2;
            ImagePlus rgb,hsb,lab;
            ImageStack stack = new ImageStack(image.getWidth(),image.getHeight());
            ColorSpaceConverter converter = new ColorSpaceConverter();
            for(int i=0;i<channels.size();++i){
                if(channels.get(i).toString()=="Red"||channels.get(i).toString()=="Blue"||channels.get(i).toString()=="Green"){
                    rgbactive=true;
                }else if(channels.get(i).toString()=="Lightness"||channels.get(i).toString()=="a"||channels.get(i).toString()=="b"){
                    labactive=true;
                }else if(channels.get(i).toString()=="Hue"||channels.get(i).toString()=="Saturation"||channels.get(i).toString()=="Brightness"){
                    hsbactive=true;
                }
            }
            if(labactive){
                lab = converter.RGBToLab(new ImagePlus("Lab", image.getStack().getProcessor(slice)));
            }else {
                lab = null;
            }
            if(rgbactive) {
                rgb = new ImagePlus("RGB",image.getStack().getProcessor(slice));
                ic = new ImageConverter(rgb);
                ic.convertToRGBStack();
            }else {
                rgb = null;
            }
            if(hsbactive) {
                hsb = new ImagePlus("HSB",image.getStack().getProcessor(slice));
                ic2 = new ImageConverter(hsb);
                ic2.convertToHSB();
            }else {
                hsb = null;
            }
            for(int i=0;i<channels.size();++i){
                switch (channels.get(i)){
                    case Lightness:
                        stack.addSlice("L",lab.getStack().getProcessor(1));
                        break;
                    case a:
                        stack.addSlice("a", lab.getStack().getProcessor(2));
                        break;
                    case b:
                        stack.addSlice("b",lab.getStack().getProcessor(3));
                        break;
                    case Red:
                        stack.addSlice("Red",rgb.getStack().getProcessor(1).convertToFloatProcessor());
                        break;
                    case Green:
                        stack.addSlice("Green",rgb.getStack().getProcessor(2).convertToFloatProcessor());
                        break;
                    case Blue:
                        stack.addSlice("Blue",rgb.getStack().getProcessor(3).convertToFloatProcessor());
                        break;
                    case Hue:
                        stack.addSlice("Hue",hsb.getStack().getProcessor(1).convertToFloatProcessor());
                        break;
                    case Saturation:
                        stack.addSlice("Saturation",hsb.getStack().getProcessor(2).convertToFloatProcessor());
                        break;
                    case Brightness:
                        stack.addSlice("Brightness",hsb.getStack().getProcessor(3).convertToFloatProcessor());
                        break;
                }
            }
            FeatureStack features = new FeatureStack(stack.getWidth(),stack.getHeight(),false);
            features.setStack(stack);

            featureStackArray.set(features,slice-1);
            // Create instances
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            for (int i=1; i<=featureStackArray.get(slice-1).getSize(); i++)
            {
            	String attString = featureStackArray.get(slice-1).getSliceLabel(i);
            	attributes.add( new Attribute( attString ) );
            }

            featuresInstances =
            		new Instances( image.getShortTitle() + "-features",
            				attributes, 1 );

            ArrayList<Point> positions = new ArrayList<Point>();
            for(int x=0;x<image.getWidth();++x){
                for(int y=0;y<image.getHeight();++y){
                    positions.add(new Point(x,y));
                }
            }
            Collections.shuffle(positions);
            for(int i=0;i<samplesPerSlice;++i){
                featuresInstances.add(featureStackArray.get(slice-1).createInstance(positions.get(i).x,positions.get(i).y));
                //IJ.log("Position:"+positions.get(i).toString()+";"+ featureStackArray.get(slice-1).createInstance(positions.get(i).x,positions.get(i).y).toString());//this values are wrong
                //IJ.log("Added element "+i+" from coordinates "+positions.get(i).x+","+positions.get(i).y);
            }
        }
    }

    /**
     * Create FeatureStackArray based on provided image and selected channels (private variable)
     * @param image input color image
     * @return feature stack array
     */
    public FeatureStackArray createFSArray(ImagePlus image){
        FeatureStackArray fsa = new FeatureStackArray(image.getStackSize());
        for(int slice = 1; slice <= image.getStackSize(); ++slice) {
            boolean labactive = false, rgbactive = false, hsbactive = false;
            ImageConverter ic, ic2;
            ImagePlus rgb, hsb, lab;
            ImageStack stack = new ImageStack(image.getWidth(), image.getHeight());
            ColorSpaceConverter converter = new ColorSpaceConverter();
            for (int i = 0; i < channels.size(); ++i) {
                if (channels.get(i).toString() == "Red" || channels.get(i).toString() == "Blue" || channels.get(i).toString() == "Green") {
                    rgbactive = true;
                } else if (channels.get(i).toString() == "Lightness" || channels.get(i).toString() == "a" || channels.get(i).toString() == "b") {
                    labactive = true;
                } else if (channels.get(i).toString() == "Hue" || channels.get(i).toString() == "Saturation" || channels.get(i).toString() == "Brightness") {
                    hsbactive = true;
                }
            }
            if (labactive) {
                lab = converter.RGBToLab(new ImagePlus("Lab", image.getStack().getProcessor(slice)));
            } else {
                lab = null;
            }
            if (rgbactive) {
                rgb = new ImagePlus("RGB",image.getStack().getProcessor(slice));
                ic = new ImageConverter(rgb);
                ic.convertToRGBStack();
            } else {
                rgb = null;
            }
            if (hsbactive) {
                hsb = new ImagePlus("HSB",image.getStack().getProcessor(slice));
                ic2 = new ImageConverter(hsb);
                ic2.convertToHSB();
            } else {
                hsb = null;
            }
            for (int i = 0; i < channels.size(); ++i) {
                switch (channels.get(i)) {
                    case Lightness:
                        stack.addSlice("L", lab.getStack().getProcessor(1));
                        break;
                    case a:
                        stack.addSlice("a", lab.getStack().getProcessor(2));
                        break;
                    case b:
                        stack.addSlice("b", lab.getStack().getProcessor(3));
                        break;
                    case Red:
                        stack.addSlice("Red", rgb.getStack().getProcessor(1).convertToFloatProcessor());
                        break;
                    case Green:
                        stack.addSlice("Green", rgb.getStack().getProcessor(2).convertToFloatProcessor());
                        break;
                    case Blue:
                        stack.addSlice("Blue", rgb.getStack().getProcessor(3).convertToFloatProcessor());
                        break;
                    case Hue:
                        stack.addSlice("Hue", hsb.getStack().getProcessor(1).convertToFloatProcessor());
                        break;
                    case Saturation:
                        stack.addSlice("Saturation", hsb.getStack().getProcessor(2).convertToFloatProcessor());
                        break;
                    case Brightness:
                        stack.addSlice("Brightness", hsb.getStack().getProcessor(3).convertToFloatProcessor());
                        break;
                }
            }
            FeatureStack features = new FeatureStack(stack.getWidth(), stack.getHeight(), false);
            features.setStack(stack);
            fsa.set(features, slice - 1);
        }
        return fsa;
    }

    /**
     * Create probability map based on provided features and current cluster.
     * @param featureStackArray input array of feature stacks
     * @return image containing the probability maps of each cluster
     */
    public ImagePlus createProbabilityMaps(FeatureStackArray featureStackArray){
        int height;
        int width;
        int numInstances;
        int numClusters=0;
        try {
            numClusters = theClusterer.numberOfClusters();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Calibration calibration = new Calibration();
        calibration = image.getCalibration();
        height = featureStackArray.getHeight();
        width = featureStackArray.getWidth();
        numInstances = height*width;
        ImageStack clusteringResult = new ImageStack(width,height);
        for(int slice = 1; slice <= featureStackArray.getSize(); ++slice){
            double clusterArray[][] = new double[numClusters][numInstances];
            FeatureStack features = featureStackArray.get(slice-1);
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            Instances instances;
            for (int i=1; i<=featureStackArray.get(slice-1).getSize(); i++)
            {
                String attString = featureStackArray.get(slice-1).getSliceLabel(i);
                attributes.add( new Attribute( attString ) );
            }

            if( featureStackArray.get(slice-1).useNeighborhood() )
                for (int i=0; i<8; i++)
                {
                    //IJ.log("Adding extra attribute original_neighbor_" + (i+1) + "...");
                    attributes.add( new Attribute( new String( "original_neighbor_" + (i+1) ) ) );
                }
            instances = new Instances(image.getTitle()+"-features", attributes, 1);
            final double[] values = new double[features.getSize()];
            final ReusableDenseInstance ins = new ReusableDenseInstance(1.0,values);
            ins.setDataset(instances);
            String[] classLabels = new String[numClusters];
            for(int i=0;i<numClusters;++i){
                classLabels[i]="Cluster "+i;
            }

            for (int x=0;x<width;++x){
                for(int y=0;y<height;++y){
                    features.setInstance(x,y,ins,values);
                    try {
                        double[] prob = theClusterer.distributionForInstance( ins );
                        for(int k = 0 ; k < numClusters; k++)
                            clusterArray[k][x+y*width] =  prob[k];
                        //IJ.log(ins.toString());
                        //IJ.log("Coordinates: "+x+","+y+" Cluster: "+clusterArray[x+y*width]);
                    }catch (Exception e){
                        e.printStackTrace();
                        return null;
                    }
                }
            }
            for(int k = 0 ; k < numClusters; k++){
                FloatProcessor processor = new FloatProcessor(width,height,clusterArray[k]);
                try {
                    processor.setMinAndMax(0,1);
                } catch (Exception e) {
                    IJ.log("Error when setting histogram range in slice: "+slice);
                }
                clusteringResult.addSlice(classLabels[k],processor);

            }

        }
        ImagePlus result = new ImagePlus("Probability map image", clusteringResult);
        result.setCalibration(calibration);
        result.setDimensions(numClusters,image.getNSlices(),image.getNFrames());
        if(image.getNSlices()*image.getNFrames()>1){
            result.setOpenAsHyperStack(true);
        }
        return result;
    }

    /**
     * Creates clustered image based on provided FeatureStackArray and using private clusterer, returns as ImagePlus
     * @param featureStackArray input array of feature stacks
     * @return clusterized image (one label per cluster)
     */
    public ImagePlus createClusteredImage(FeatureStackArray featureStackArray){
        int height;
        int width;
        int numInstances;
        Calibration calibration = new Calibration();
        calibration = image.getCalibration();
        height = featureStackArray.getHeight();
        width = featureStackArray.getWidth();
        numInstances = height*width;
        ImageStack clusteringResult = new ImageStack(width,height);
        for(int slice = 1; slice <= featureStackArray.getSize(); ++slice){
            byte clusterArray[] = new byte[numInstances];
            FeatureStack features = featureStackArray.get(slice-1);
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            Instances instances;
            for (int i=1; i<=featureStackArray.get(slice-1).getSize(); i++)
            {
                String attString = featureStackArray.get(slice-1).getSliceLabel(i);
                attributes.add( new Attribute( attString ) );
            }

            instances = new Instances(image.getShortTitle()+"-features", attributes, 1);
            final double[] values = new double[features.getSize()];
            final ReusableDenseInstance ins = new ReusableDenseInstance(1.0,values);
            ins.setDataset(instances);
            for (int x=0;x<width;++x){
                for(int y=0;y<height;++y){
                    features.setInstance(x,y,ins,values);
                    try {
                        clusterArray[x+y*width]= (byte) theClusterer.clusterInstance(ins);
                        //IJ.log(ins.toString());
                        //IJ.log("Coordinates: "+x+","+y+" Cluster: "+clusterArray[x+y*width]);
                    }catch (Exception e){
                        e.printStackTrace();
                        return null;
                    }
                }
            }
            ByteProcessor processor = new ByteProcessor(width,height,clusterArray);
            try {
                processor.setMinAndMax(0,theClusterer.numberOfClusters());
            } catch (Exception e) {
            	e.printStackTrace();
                return null;
            }
            clusteringResult.addSlice(processor);
        }
        ImagePlus result = new ImagePlus("Clustered image", clusteringResult);
        result.setCalibration(calibration);
        return result;
    }


    /**
     * Create ARFF file of features with provided name.
     * @param path complete path to output ARFF file
     * @param theInstances dataset of instances to saved to file
     * @return true if the file was correctly saved, false otherwise
     */
    public boolean createFile(
    		String path,
    		Instances theInstances )
    {
    	boolean saved = false;

        if( path != null)
        {
            File file = new File( path );

            BufferedWriter out = null;
            try{
                out = new BufferedWriter(
                        new OutputStreamWriter(
                                new FileOutputStream( file), StandardCharsets.UTF_8 ) );

                final Instances header = new Instances(theInstances, 0);
                out.write(header.toString());

                for(int i = 0; i < theInstances.numInstances(); i++)
                {
                    out.write(theInstances.get(i).toString()+"\n");
                }
                saved = true;
            }
            catch(Exception e)
            {
            	saved = false;
                IJ.log("Error: couldn't write instances into .ARFF file.");
                IJ.showMessage("Exception while saving data as ARFF file");
                e.printStackTrace();
            }
            finally{
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }else {
            IJ.log("Error when choosing path");
        }
        return saved;
    }

    /**
     * Save clusterer to a file
     * @param filename path to file to be saved
     * @return boolean value showing success/failure
     */
    public boolean saveClusterer(String filename){
        File sFile = null;
        boolean saveOK = true;


        IJ.log("Saving model to file...");

        try {
            sFile = new File(filename);
            OutputStream os = new FileOutputStream(sFile);
            if (sFile.getName().endsWith(".gz"))
            {
                os = new GZIPOutputStream(os);
            }
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject(theClusterer);
            Instances trainHeader = new Instances(featuresInstances,0);
            objectOutputStream.writeObject(trainHeader);
            objectOutputStream.flush();
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            IJ.error("Save Failed", "Error when saving classifier into a file");
            saveOK = false;
        }
        if (saveOK)
            IJ.log("Saved model into " + filename );

        return saveOK;
    }

    /**
     * Load clusterer from path
     * @param path Path of file to be loaded
     * @return boolean value showing success/failure
     */
    public boolean loadClusterer(String path){

        File selected = new File(path);

        try{
            InputStream is = new FileInputStream(selected);
            if(selected.getName().endsWith(".gz")){
                is = new GZIPInputStream(is);
            }
            ObjectInputStream objectInputStream = SerializationHelper.getObjectInputStream(is);
            AbstractClusterer clusterer =
            		(AbstractClusterer) objectInputStream.readObject();
            if( null == clusterer )
            	return false;
            Instances ins = (Instances) objectInputStream.readObject();
            if( null == ins )
            	return false;
            theClusterer = clusterer;
            featuresInstances = ins;
            objectInputStream.close();
        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        return true;
    }


    //Getters and setters

    /**
     * Get number of samples that is being used
     * @return number of samples that is being used
     */
    public int getNumSamples() {
        return numSamples;
    }

    /**
     * Set number of samples that is to be used
     * @param numSamples number of samples that is to be used
     */
    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    /**
     * Get image that is being used
     * @return color image being used
     */
    public ImagePlus getImage() {
        return image;
    }

    /**
     * Set image that is to be used
     * @param image color image to be clusterized
     */
    public void setImage(ImagePlus image) {
        this.image = image;
    }

    /**
     * Get FeatureStackArray that is being used
     * @return array of feature stacks used for clustering
     */
    public FeatureStackArray getFeatureStackArray() {
        return featureStackArray;
    }

    /**
     * Set FeatureStackArray that is to be used
     * @param featureStackArray array of feature stacks to use in clustering
     */
    public void setFeatureStackArray(FeatureStackArray featureStackArray) {
        this.featureStackArray = featureStackArray;
    }


    /**
     * Get features instances that are being used
     * @return set of instances used for building the clusterer
     */
    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    /**
     * Set features instances that are to be used
     * @param featuresInstances set of instances to be used to build the clusterer
     */
    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }


    /**
     * Get abstract clusterer that has been created
     * @return current clusterer
     */
    public AbstractClusterer getTheClusterer() {
        return theClusterer;
    }

    /**
     * Set abstract clusterer to be used
     * @param theClusterer clustering Weka model to use
     */
    public void setTheClusterer(AbstractClusterer theClusterer) {
        this.theClusterer = theClusterer;
    }
    /**
     * Set channels to use
     * @param selectedChannels list of selected channels
     */
    public void setChannels( ArrayList<Channel> selectedChannels )
    {
       this.channels = selectedChannels;
    }
}

