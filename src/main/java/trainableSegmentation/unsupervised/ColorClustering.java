package trainableSegmentation.unsupervised;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ColorSpaceConverter;
import ij.process.ImageProcessor;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by 96jsa on 17/06/15.
 */
public class ColorClustering {

    private Instances featuresInstances;
    private ImagePlus image;
    private FeatureStackArray featureStackArray;
    private int numSamples;

    /**
     * Creates features based on image and number of samples
     * @param image
     * @param numSamples
     */
    public ColorClustering(ImagePlus image, int numSamples){
        this.setImage(image);
        this.setNumSamples(numSamples);
        featureStackArray = new FeatureStackArray(image.getStackSize());
        this.createFeatures();
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
            for(int i=0;i<numSamples;++i){ //Problem: When choosing points they can repeat, so you may have duplicates
                int randx = rand.nextInt((image.getWidth()-1-0)+1)+0;//(max-min+1)+min
                int randy = rand.nextInt((image.getHeight()-1-0)+1)+0;//(max-min+1)+min
                featuresInstances.add(featureStackArray.get(slice-1).createInstance(randx,randy));
            }
        }
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
