package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileInfo;
import ij.measure.Calibration;
import ij.plugin.Converter;
import ij.process.*;
import org.apache.commons.math3.analysis.function.Abs;
import sun.security.jca.GetInstance;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.ReusableDenseInstance;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.PropertyPath;
import weka.core.stopwords.Null;

import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Optional;
import java.util.Random;


public class ColorClustering {


    /**
     * Names all channels that can be used.
     * */
    public enum Channel{
        Red("Red"),
        Green("Green"),
        Blue("Blue"),
        Lightness("Lightness"),
        a("a"),
        b("b"),
        Hue("Hue"),
        Saturation("Saturation"),
        Brightness("Brightness");

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
         * @return
         */
        public String toString(){
            return this.label;
        }

        /**
         * Returns total number of channels, static method.
         * @return
         */
        public static int numChannels(){
            int number=0;
            for(String item : getAllLabels()){
                number++;
            }
            return number;
        }

        /**
         * Get all labels in a String[] structure.
         * @return
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
         * @return
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

    private ArrayList<Channel> channels = new ArrayList<Channel>();
    private Instances featuresInstances;
    private ImagePlus image;
    private FeatureStackArray featureStackArray;
    private int numSamples;
    private AbstractClusterer theClusterer;


    /**
     * Creates color clustering object based on image, number of samples and selected channels. Creates the features.
     * @param image
     * @param numSamples
     * @param selectedChannels
     */
    public ColorClustering(ImagePlus image, int numSamples, ArrayList<Channel> selectedChannels){ //Separar build clusterer del constructor
        for(Channel element: selectedChannels){
            this.channels.add(element);
        }
        this.setImage(image);
        this.setNumSamples(numSamples);
        featureStackArray = new FeatureStackArray(image.getStackSize());
        this.createFeatures();
    }

    /**
     * Create clusterer using features, and provided number of clusters;
     * @param numClusters
     * @return
     */
    public AbstractClusterer createClusterer(AbstractClusterer selectedClusterer){
        PixelClustering pixelClustering = new PixelClustering(this.getFeaturesInstances(),selectedClusterer);

            pixelClustering.buildClusterer();
        AbstractClusterer clusterer = pixelClustering.getSelectedClusterer();
        return clusterer;
    }


    /**
     * Creates features based on selected channels, uses private variables.
     */
    public void createFeatures(){
        IJ.log("Creating Features");
        int numSlices = image.getNSlices();
        int samplesPerSlice = numSamples/numSlices;
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
            if( null == featuresInstances )
            {
                IJ.log("Initializing loaded data...");
                // Create instances
                ArrayList<Attribute> attributes = new ArrayList<Attribute>();
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
                featuresInstances = new Instances("segment", attributes, 1);
            }
            Random rand = new Random();
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
     * Creates FeatureStackArray based on provided image and selected channels (private variable)
     * @param image
     * @return
     */
    public FeatureStackArray createFSArray(ImagePlus image){
        IJ.log("Creating Feature Stack Array");
        int height;
        int width;
        int numInstances;
        height = image.getHeight();
        width = image.getWidth();
        numInstances = height*width;
        FeatureStackArray theFeatures = new FeatureStackArray(image.getStackSize());
        ImageStack clusteringResult = new ImageStack(width,height);
        double clusterArray[] = new double[numInstances];
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
            theFeatures.set(features, slice - 1);
        }
        return theFeatures;
    }

    /**
     * Creates probability map based on provided features
     * @param theFeatures
     * @return
     */
    public ImagePlus createProbabilityMaps(FeatureStackArray theFeatures){
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
        height = theFeatures.getHeight();
        width = theFeatures.getWidth();
        numInstances = height*width;
        ImageStack clusteringResult = new ImageStack(width,height);
        for(int slice = 1; slice <= theFeatures.getSize(); ++slice){
            double clusterArray[][] = new double[numClusters][numInstances];
            FeatureStack features = theFeatures.get(slice-1);
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            Instances instances;
            for (int i=1; i<=theFeatures.get(slice-1).getSize(); i++)
            {
                String attString = theFeatures.get(slice-1).getSliceLabel(i);
                attributes.add( new Attribute( attString ) );
            }

            if( theFeatures.get(slice-1).useNeighborhood() )
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
                        IJ.log("Error when applying clusterer to pixel: "+x+","+y);
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
     * @param theFeatures
     * @return
     */
    public ImagePlus createClusteredImage(FeatureStackArray theFeatures){
        IJ.log("Creating clustered Image");
        int height;
        int width;
        int numInstances;
        Calibration calibration = new Calibration();
        calibration = image.getCalibration();
        height = theFeatures.getHeight();
        width = theFeatures.getWidth();
        numInstances = height*width;
        ImageStack clusteringResult = new ImageStack(width,height);
        for(int slice = 1; slice <= theFeatures.getSize(); ++slice){
            byte clusterArray[] = new byte[numInstances];
            FeatureStack features = theFeatures.get(slice-1);
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            Instances instances;
            for (int i=1; i<=theFeatures.get(slice-1).getSize(); i++)
            {
                String attString = theFeatures.get(slice-1).getSliceLabel(i);
                attributes.add( new Attribute( attString ) );
            }

            if( theFeatures.get(slice-1).useNeighborhood() )
                for (int i=0; i<8; i++)
                {
                    //IJ.log("Adding extra attribute original_neighbor_" + (i+1) + "...");
                    attributes.add( new Attribute( new String( "original_neighbor_" + (i+1) ) ) );
                }
            instances = new Instances(image.getTitle()+"-features", attributes, 1);
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
                        IJ.log("Error when applying clusterer to pixel: "+x+","+y);
                    }
                }
            }
            ByteProcessor processor = new ByteProcessor(width,height,clusterArray);
            try {
                processor.setMinAndMax(0,theClusterer.numberOfClusters());
            } catch (Exception e) {
                IJ.log("Error when setting histogram range in slice: "+slice);
            }
            clusteringResult.addSlice(processor);
        }
        ImagePlus result = new ImagePlus("Clustered image", clusteringResult);
        result.setCalibration(calibration);
        return result;
    }


    /**
     * Creates file of features with provided name.
     * @param name
     * @param theInstances
     */
    public void createFile(Instances theInstances){

        String path = IJ.getFilePath("Choose a path to save the arff file to");
        if(path!=null){
            File file = new File(path+".arff");
            IJ.log("Creating file");
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
                IJ.log("Created file");
            }
            catch(Exception e)
            {
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

    }



    //Getters and setters

    /**
     * Get number of samples that is being used
     * @return
     */
    public int getNumSamples() {
        return numSamples;
    }

    /**
     * Set number of samples that is to be used
     * @param numSamples
     */
    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    /**
     * Get image that is being used
     * @return
     */
    public ImagePlus getImage() {
        return image;
    }

    /**
     * Set image that is to be used
     * @param image
     */
    public void setImage(ImagePlus image) {
        this.image = image;
    }

    /**
     * Get FeatureStackArray that is being used
     * @return
     */
    public FeatureStackArray getFeatureStackArray() {
        return featureStackArray;
    }

    /**
     * Set FeatureStackArray that is to be used
     * @param featureStackArray
     */
    public void setFeatureStackArray(FeatureStackArray featureStackArray) {
        this.featureStackArray = featureStackArray;
    }


    /**
     * Get features instances that are being used
     * @return
     */
    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    /**
     * Set features instances that are to be used
     * @param featuresInstances
     */
    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }


    /**
     * Get abstract clusterer that has been created
     * @return
     */
    public AbstractClusterer getTheClusterer() {
        return theClusterer;
    }

    /**
     * Set abstract clusterer to be used
     * @param theClusterer
     */
    public void setTheClusterer(AbstractClusterer theClusterer) {
        this.theClusterer = theClusterer;
    }

}

