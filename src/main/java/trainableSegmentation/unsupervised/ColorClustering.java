package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ColorSpaceConverter;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.apache.commons.math3.analysis.function.Abs;
import sun.security.jca.GetInstance;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.ReusableDenseInstance;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Comments need to be updated
 */
public class ColorClustering {

    private Instances featuresInstances;
    private ImagePlus image;
    private FeatureStackArray featureStackArray;
    private int numSamples;
    private AbstractClusterer theClusterer;

    /**
     * Creates features based on image and number of samples
     * @param image
     * @param numSamples
     */
    public ColorClustering(ImagePlus image, int numSamples, int numClusters){
        this.setImage(image);
        this.setNumSamples(numSamples);
        featureStackArray = new FeatureStackArray(image.getStackSize());
        this.createFeatures();
        PixelClustering pixelClustering = new PixelClustering(this.getFeaturesInstances(),numClusters);
        theClusterer = pixelClustering.getClusterer();
    }

    /**
     * Creates a b features based on RGB image into Lab image, chooses numSamples pixels at random
     */
    public void createFeatures(){
        for(int slice = 1; slice <= image.getStackSize(); ++slice){
            ImageStack stack = new ImageStack(image.getWidth(),image.getHeight());

            ColorSpaceConverter converter = new ColorSpaceConverter();

            ImagePlus lab = converter.RGBToLab(new ImagePlus("RGB",image.getStack().getProcessor(slice)));

            stack.addSlice("a", lab.getStack().getProcessor(2));
            stack.addSlice("b",lab.getStack().getProcessor(3));
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
                        IJ.log("Adding extra attribute original_neighbor_" + (i+1) + "...");
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
            for(int i=0;i<numSamples;++i){
                featuresInstances.add(featureStackArray.get(slice-1).createInstance(positions.get(i).x,positions.get(i).y));
                IJ.log("Added element "+i+" from coordinates "+positions.get(i).x+","+positions.get(i).y);
            }
        }
    }


    public ImagePlus createClusteredImage(){
        theClusterer.setDebug(true);
        IJ.log(theClusterer.toString());
        int height = image.getHeight();
        int width = image.getWidth();
        int numInstances = height*width;
        ImageStack clusteringResult = new ImageStack(width, height);
        double clusterArray[] = new double[numInstances];
        FeatureStack sliceFeatures = new FeatureStack(image);
        final double[] values = new double[ sliceFeatures.getSize() + 1];
        // create empty reusable instance
        final ReusableDenseInstance ins =
                new ReusableDenseInstance( 1.0, values );
        ins.setDataset(featuresInstances);
        for(int x=0;x<width;++x){
            for(int y=0;y<height;++y){
                sliceFeatures.setInstance(x,y,0,ins,values);
                try {
                    clusterArray[x+y*width]=theClusterer.clusterInstance(ins);
                    //IJ.log("assigned cluster: "+clusterArray[x+y*width]);
                } catch (Exception e) {
                    IJ.log("Error when applying clusterer to pixel: "+x+","+y);
                }
            }
        }
        clusteringResult.addSlice(new FloatProcessor(width,height,clusterArray));
        return new ImagePlus("clustered image", clusteringResult);
    }

    /**
     * Creates arff file
     * @param name name of the file to be created
     */
    public void createFile(String name){
        BufferedWriter out = null;
        try{
            out = new BufferedWriter(
                    new OutputStreamWriter(
                            new FileOutputStream( name), StandardCharsets.UTF_8 ) );

            final Instances header = new Instances(featuresInstances, 0);
            out.write(header.toString());

            for(int i = 0; i < featuresInstances.numInstances(); i++)
            {
                out.write(featuresInstances.get(i).toString()+"\n");
            }
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
        IJ.log("Created file");
    }


    //Getters and setters

    public int getNumSamples() {
        return numSamples;
    }

    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    public ImagePlus getImage() {
        return image;
    }

    public void setImage(ImagePlus image) {
        this.image = image;
    }


    public FeatureStackArray getFeatureStackArray() {
        return featureStackArray;
    }

    public void setFeatureStackArray(FeatureStackArray featureStackArray) {
        this.featureStackArray = featureStackArray;
    }


    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }

}
