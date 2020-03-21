package net.haesleinhuepf.clijx.weka;

import hr.irb.fastRandomForest_clij.FastRandomForest;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.NewImage;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clijx.CLIJx;
import weka.classifiers.AbstractClassifier;

public class WekaDemo {
    public static void main(String[] args) {
        // Load Example data
        ImagePlus inputImp = IJ.openImage("src/test/resources/NPC_T01_c2.tif");
        IJ.run(inputImp, "32-bit", "");
        ImagePlus partialGroundTruthImp = IJ.openImage("src/test/resources/NPC_T01_c2_ground_truth.tif");

        // init GPU
        CLIJx clijx = CLIJx.getInstance();

        // push to GPU
        ClearCLBuffer input = clijx.push(inputImp);
        ClearCLBuffer partialGroundTruth = clijx.push(partialGroundTruthImp);
        ClearCLBuffer featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, "original gaussianblur=1 gaussianblur=5 sobelofgaussian=1 sobelofgaussian=5");

        new ImageJ();

        // show input data
        clijx.show(input, "input");
        clijx.show(partialGroundTruth, "partial ground truth");
        clijx.show(featureStack, "feature stack");

        // Train (internally with Fijis Trainable Segmentation)
        CLIJxWeka cw = new CLIJxWeka(clijx, featureStack, partialGroundTruth);

        FastRandomForest classifier = cw.getClassifier();
        int numberOfClasses = cw.getNumberOfClasses();

        // Predict (internally with Fijis Trainable Segmentation)
        CLIJxWeka cw2 = new CLIJxWeka(clijx, featureStack, classifier, numberOfClasses);

        ClearCLBuffer buffer = cw2.getClassification();
        clijx.show(buffer, "classification");

        // Predict (internally with OpenCL)
        String openCLkernelcode = cw2.getOCL();
        ApplyOCLWekaModel.applyOCL(clijx, featureStack, buffer, openCLkernelcode);
        clijx.show(buffer, "classification_opencl");












        clijx.clear();
















    }
}
