package net.haesleinhuepf.clijx.weka;

import hr.irb.fastRandomForest_clij.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.process.FloatProcessor;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clijx.CLIJx;
import net.imglib2.img.array.ArrayImgs;
import trainableSegmentation_clij.WekaSegmentation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * CLIJxWeka
 *
 * Builds the bridge between Fijis Trainable Weka Segmentation and CLIJ
 * https://github.com/fiji/Trainable_Segmentation
 * https://clij.github.io
 *
 * Author: Robert Haase, MPI CBG Dresden, rhaase@mpi-cbg.de
 *
 * Parts of the code here were copied over from the Trainable_Segmentation repository (link above). Thus,
 * this code is licensed GPL2 as well.
 *
 *  License: GPL
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License 2
 *  as published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *  Authors: Verena Kaynig (verena.kaynig@inf.ethz.ch), Ignacio Arganda-Carreras (iarganda@mit.edu)
 *           Albert Cardona (acardona@ini.phys.ethz.ch)
 *
 */
public class CLIJxWeka {


    private FastRandomForest classifier;
    private Integer numberOfClasses;
    private Integer numberOfFeatures;
    private CLIJx clijx;
    private ClearCLBuffer featureStack;
    private ClearCLBuffer classification;
    private String oclCode;


    private int frf_numberOfTrees = 200;
    private int frf_maxDepth = 0;
    private int frf_numberOfFeatures = 2;

    public CLIJxWeka(CLIJx clijx, ClearCLBuffer featureStack, ClearCLBuffer classification) {
        this.clijx = clijx;
        this.featureStack = featureStack;
        this.classification = classification;
    }

    public CLIJxWeka(CLIJx clijx, ClearCLBuffer featureStack, FastRandomForest classifier, Integer numberOfClasses) {
        this.clijx = clijx;
        this.featureStack = featureStack;
        this.classifier = classifier;
        this.numberOfClasses = numberOfClasses;
        numberOfFeatures = (int)featureStack.getDepth();
        this.oclCode = classifier.translateToOcl(numberOfClasses, numberOfFeatures);
    }

    public CLIJxWeka(CLIJx clijx, ClearCLBuffer featureStack, String classifierFilename) {
        this.clijx = clijx;
        this.featureStack = featureStack;
        loadClassifier(classifierFilename);
    }

    private void trainClassifier() {
        if (classifier != null) {
            System.out.println("Already trained.");
            return;
        }
        if (classification == null) {
            System.out.println("No ground truth available");
            return;
        }

        numberOfClasses = (int) clijx.maximumOfAllPixels(classification); // background 0 doesn't count as class
        numberOfFeatures = (int)featureStack.getDepth();
        ArrayList<Attribute> attributes = makeAttributes(numberOfClasses, numberOfFeatures);

        System.out.println("att size" + attributes.size());

        Instances trainingData =  new Instances( "segment", attributes, 1 );
        // Set the index of the class attribute
        trainingData.setClassIndex(attributes.size() - 1);

        // convert features and classification ground truth to instances
        featureStackToInstance(clijx, featureStack, classification, trainingData);

        System.out.println("Balance training data");
        System.out.println("Num classes " + trainingData.numClasses());

        // not sure if this is necessary
        trainingData = WekaSegmentation.balanceTrainingData(trainingData);



        System.out.println("Init classifier");


        // Initialization of Fast Random Forest classifier
        FastRandomForest classifier = new FastRandomForest();
        classifier.setNumTrees(frf_numberOfTrees);

        // Random seed
        classifier.setSeed( (new Random()).nextInt() );

        //this is the default that Breiman suggests
        //rf.setNumFeatures((int) Math.round(Math.sqrt(featureStack.getSize())));
        //but this seems to work better
        classifier.setNumFeatures(frf_numberOfFeatures);

        classifier.setNumThreads( Prefs.getThreads() );
        classifier.setMaxDepth(frf_maxDepth);

        System.out.println("Train classifier");

        // Train the classifier on the current data
        try{
            classifier.buildClassifier(trainingData);
        }
        catch (InterruptedException ie)
        {
            IJ.log("Classifier construction was interrupted.");
        }
        catch(Exception e){
            IJ.showMessage(e.getMessage());
            e.printStackTrace();
        }

        oclCode = classifier.translateToOcl(numberOfClasses, numberOfFeatures);

        // Print classifier information
        IJ.log( classifier.toString() );



        System.out.println("Evaluate classifier on training data");

        double error = -1;
        try {
            final Evaluation evaluation = new Evaluation(trainingData);
            evaluation.evaluateModel(classifier, trainingData);

            System.out.println(evaluation.toSummaryString("\n=== Test data evaluation ===\n", false));
            System.out.println(evaluation.toClassDetailsString() + "\n");
            System.out.println(evaluation.toMatrixString());

            error = evaluation.errorRate();
        } catch (Exception e) {
            e.printStackTrace();
        }

        this.classifier = classifier;
        this.numberOfClasses = numberOfClasses;
    }

    private static ArrayList<Attribute> makeAttributes(int numberOfClasses, int numberOfFeatures) {
        System.out.println("Number of classes: " + numberOfClasses);
        ArrayList<String> classes = new ArrayList<>();
        for (int i = 0; i < numberOfClasses; i++) {
            classes.add("C" + (i + 1));
        }
        System.out.println("Classes: " + classes.size());

        // add features (represented by slices
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < numberOfFeatures; i++ ) {
            attributes.add(new Attribute("F" + (i + 1)));
        }

        // add training ground truth
        attributes.add(new Attribute("class", classes));

        return attributes;
    }

    private static void featureStackToInstance(CLIJx clijx, ClearCLBuffer stack, ClearCLBuffer classification, Instances instances) {
        // transpose stack for faster access in feature (Z) direction
        // and convert to float
        ClearCLBuffer transposed = clijx.create(new long[]{stack.getDepth(), stack.getHeight(), stack.getWidth()}, clijx.Float);
        clijx.transposeXZ(stack, transposed);

        ClearCLBuffer classificationConverted = classification;
        if (classificationConverted.getNativeType() != clijx.Float) {
            classificationConverted = clijx.create(new long[]{classification.getWidth(), classification.getHeight()}, clijx.Float);
            clijx.copy(classification, classificationConverted);
        }

        ImagePlus features = clijx.pull(transposed);
        ImagePlus classified = clijx.pull(classification);

        float[] classes = (float[]) classified.getProcessor().getPixels();
        //System.out.println("ground truth: " + Arrays.toString(classes));


        int numberOfFeatures = (int) stack.getDepth();
        int width = (int) stack.getWidth();
        int height = (int) stack.getHeight();

        System.out.println("Number of features: " + numberOfFeatures);

        for (int x = 0; x < width; x++) {
            features.setZ(x + 1); // the feature stack is XZ - transposed; its Z corresponds to original image width

            float[] pixels = (float[]) features.getProcessor().getPixels();
            // see how pixels are addressed here: ((FloatProcessor)features.getProcessor()).getPixel(1,1)
            for (int y = 0; y < height; y++) {
                if (classes[y * width + x] != 0) {
                    double[] values = new double[numberOfFeatures + 1]; // number of features + ground truth
                    for (int f = 0; f < numberOfFeatures; f++) {
                        values[f] = pixels[y * numberOfFeatures + f];
                    }
                    values[values.length - 1] = classes[y * width + x] - 1; // minus 1 because background isn't evaluated
                    //System.out.println("inst: " + Arrays.toString(values));
                    instances.add(new DenseInstance(1.0, values));
                }
            }
        }

        System.out.println("number of instances: " + instances.size());

        if (classification != classificationConverted) {
            clijx.release(classificationConverted);
        }

        clijx.release(transposed);
    }

    private static ClearCLBuffer featureStackToInstance(CLIJx clijx, ClearCLBuffer stack, AbstractClassifier classifier, int numberOfClasses) {
        // transpose stack for faster access in feature (Z) direction
        // and convert to float
        ClearCLBuffer transposed = clijx.create(new long[]{stack.getDepth(), stack.getHeight(), stack.getWidth()}, clijx.Float);
        clijx.transposeXZ(stack, transposed);

        ImagePlus features = clijx.pull(transposed);
        ImagePlus classified = new ImagePlus("classified", new FloatProcessor((int)stack.getWidth(), (int)stack.getHeight()));
        //clijx.pull(classification);

        float[] classes = (float[]) classified.getProcessor().getPixels();
        //System.out.println("ground truth: " + Arrays.toString(classes));


        int numberOfFeatures = (int) stack.getDepth();
        int width = (int) stack.getWidth();
        int height = (int) stack.getHeight();

        ArrayList<Attribute> attributes = makeAttributes(numberOfClasses, numberOfFeatures);
        Instances dataSet = new Instances( "segment", attributes, 1 );
        dataSet.setClassIndex(attributes.size() - 1);

        System.out.println("Hello1");
        for (int x = 0; x < width; x++) {
            features.setZ(x + 1); // the feature stack is XZ - transposed; its Z corresponds to original image width

            float[] pixels = (float[]) features.getProcessor().getPixels();
            // see how pixels are addressed here: ((FloatProcessor)features.getProcessor()).getPixel(1,1)
            for (int y = 0; y < height; y++) {
                double[] values = new double[numberOfFeatures + 1]; // number of features + ground truth
                for (int f = 0; f < numberOfFeatures; f++) {
                    values[f] = pixels[y * numberOfFeatures + f];
                }
                //values[values.length - 1] = classes[y * width + x] - 1; // minus 1 because background isn't evaluated
                //System.out.println("inst: " + Arrays.toString(values));
                Instance instance = new DenseInstance(1.0, values);
                instance.setDataset(dataSet);
                try {
                    float klass = (float)classifier.classifyInstance(instance) + 1; // plus 1 because background isn't evaluated.
                    classes[y * width + x] = klass;
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }
        clijx.release(transposed);

        return clijx.push(classified);
    }


    private void applyClassifier() {
        if (classification != null) {
            System.out.println("Alread classified");
            return;
        }
        if (classifier == null) {
            System.out.println("No classifier available.");
            return;
        }

        classification = featureStackToInstance(clijx, featureStack, classifier, numberOfClasses);
    }

    public FastRandomForest getClassifier() {
        trainClassifier();
        return classifier;
    }

    public ClearCLBuffer getClassification() {
        applyClassifier();
        return classification;
    }

    /*
    public ClearCLBuffer getClassificationViaOcl() {
        oclCode
        classification = featureStackToInstance(clijx, featureStack, classifier, numberOfClasses);

        if (new File(loadModelFilename + ".cl").exists()) {
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("src_featureStack", srcFeatureStack3D);
            parameters.put("dst", dstClassificationResult);
            parameters.put("export_probabilities", 0);
            clijx.execute(Object.class,loadModelFilename + ".cl", "classify_feature_stack", dstClassificationResult.getDimensions(), dstClassificationResult.getDimensions(), parameters);
        } else {
            new IllegalArgumentException("This model hasn't been saved as OCL Model. Try applyWekaModel instead.");
        }
    }
    */

    public void saveClassifier(String filename) {
        if (classifier == null) {
            trainClassifier();
        }
        if (classifier == null) {
            System.out.println("No classifier to save");
            return;
        }
        if (new File(filename).getParentFile() != null) {
            new File(filename).getParentFile().mkdirs();
        }

        File outputTarget = new File(filename + ".cl");
        try {
            FileWriter writer  = new FileWriter(outputTarget);
            writer.write(oclCode);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            File sFile = new File(filename);
            OutputStream os = new FileOutputStream(sFile);
            if (sFile.getName().endsWith(".gz"))
            {
                os = new GZIPOutputStream(os);
            }
            ObjectOutputStream oos = new ObjectOutputStream(os);
            oos.writeObject(classifier);
            oos.writeObject(numberOfClasses);
            oos.writeObject(numberOfFeatures);
            oos.flush();
            oos.close();
        }
        catch (Exception e)
        {
            IJ.error("Save Failed", "Error when saving classifier into a file");
        }
    }

    private void loadClassifier(String filename) {
        try {
            File selected = new File(filename);
            InputStream is = new FileInputStream(selected);
            if (selected.getName().endsWith(".gz")) {
                is = new GZIPInputStream(is);
            }
            ObjectInputStream ois = new ObjectInputStream(is);

            classifier = (FastRandomForest) ois.readObject();
            numberOfClasses = (Integer) ois.readObject();
            numberOfFeatures = (Integer) ois.readObject();


            oclCode = classifier.translateToOcl(numberOfClasses, numberOfFeatures);

            ois.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public Integer getNumberOfClasses() {
        return numberOfClasses;
    }

    public void printClassifier() {
        System.out.println(classifier);
    }

    public  static void main(String... args) {
        float[] f_ground_truth = {2,2,2,2,2,1,1,1,1,1};
        float[] f_measure1 =     {0,1,2,3,4,5,6,7,8,9};
        float[] f_measure2 =     {2,3,2,3,2,3,2,3,2,3};

        CLIJx clijx = CLIJx.getInstance();

        ClearCLBuffer ground_truth = clijx.push(ArrayImgs.floats(f_ground_truth, new long[]{10, 1, 1}));
        ClearCLBuffer measure1 = clijx.push(ArrayImgs.floats(f_measure1, new long[]{10, 1}));
        ClearCLBuffer measure2 = clijx.push(ArrayImgs.floats(f_measure2, new long[]{10, 1}));

        ClearCLBuffer featureStack = clijx.create(new long[]{10, 1, 2}, clijx.Float);

        // copy features into feature stack
        clijx.copySlice(measure1, featureStack, 0);
        clijx.copySlice(measure2, featureStack, 1);

        CLIJxWeka clijxweka = new CLIJxWeka(clijx, featureStack, ground_truth);
        System.out.println(clijxweka.getClassifier());

        String modelFilename = "C:/structure/models/clijxwekatest.model";

        clijxweka.saveClassifier(modelFilename);

        // test trained model:
        {
            clijxweka.classification = null;
            ClearCLBuffer result = clijxweka.getClassification();
            ImagePlus converted = clijx.pull(result);
            FloatProcessor fp = (FloatProcessor)converted.getProcessor();
            System.out.println("Applied model: " + Arrays.toString((float[]) fp.getPixels()));
        }

        // Test with model from disc
        {
            CLIJxWeka clijxweka2 = new CLIJxWeka(clijx, featureStack, modelFilename);
            ClearCLBuffer result = clijxweka2.getClassification();
            ImagePlus converted = clijx.pull(result);
            FloatProcessor fp = (FloatProcessor) converted.getProcessor();
            System.out.println("Applied model from dis: " + Arrays.toString((float[]) fp.getPixels()));
        }


        clijx.release(ground_truth);
        clijx.release(measure1);
        clijx.release(measure2);
        clijx.release(featureStack);

    }


    public String getOCL() {
        return oclCode;
    }


    public void setNumberOfTrees(int frf_numberOfTrees) {
        this.frf_numberOfTrees = frf_numberOfTrees;
    }

    public void setMaxDepth(int frf_maxDepth) {
        this.frf_maxDepth = frf_maxDepth;
    }

    public void setNumberOfFeatures(int frf_numberOfFeatures) {
        this.frf_numberOfFeatures = frf_numberOfFeatures;
    }

    public void setFeatureStack(ClearCLBuffer featureStack) {
        this.featureStack = featureStack;
        if (this.classification != null) {
            clijx.release(this.classification);
        }
        this.classification = null;
    }
}
