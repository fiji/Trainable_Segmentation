package net.haesleinhuepf.clijx.weka;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.NewImage;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clijx.CLIJx;
import weka.classifiers.AbstractClassifier;

public class WekaDemo {
    public static void main(String[] args) {
        ImagePlus inputImp = IJ.openImage("src/test/resources/blobs.tif");
        IJ.run(inputImp, "32-bit", "");
        ImagePlus partialGroundTruthImp = NewImage.createFloatImage("label map", inputImp.getWidth(), inputImp.getHeight(), 1, NewImage.FILL_BLACK);

        // true pixels
        partialGroundTruthImp.setRoi(21,51,17,13);
        partialGroundTruthImp.getProcessor().setColor(2);
        partialGroundTruthImp.getProcessor().fill();

        // false pixels
        partialGroundTruthImp.setRoi(101,37,20,16);
        partialGroundTruthImp.getProcessor().setColor(1);
        partialGroundTruthImp.getProcessor().fill();

        int numberOfFeatures = 11;

        CLIJx clijx = CLIJx.getInstance();
        ClearCLBuffer input = clijx.push(inputImp);
        ClearCLBuffer partialGroundTruth = clijx.push(partialGroundTruthImp);
        ClearCLBuffer temp = clijx.create(input);
        ClearCLBuffer temp2 = clijx.create(input);
        ClearCLBuffer featureStack = clijx.create(new long[] {input.getWidth(), input.getHeight(), numberOfFeatures});

        int featureCount = 0;

        clijx.copySlice(input, featureStack, featureCount);
        featureCount ++;


        for (int i = 0; i < 5; i++) {
            double sigma = (i + 1);
            clijx.blur(input, temp, sigma, sigma);

            clijx.sobel(temp, temp2);

            clijx.copySlice(temp, featureStack, featureCount);
            featureCount ++;
            clijx.copySlice(temp2, featureStack, featureCount);
            featureCount ++;

        }

        new ImageJ();

        clijx.show(input, "input");

        clijx.show(partialGroundTruth, "partial ground truth");

        clijx.show(featureStack, "feature stack");


        net.haesleinhuepf.clijx.weka.CLIJxWeka cw = new net.haesleinhuepf.clijx.weka.CLIJxWeka(clijx, featureStack, partialGroundTruth);

        AbstractClassifier classifier = cw.getClassifier();
        int numberOfClasses = cw.getNumberOfClasses();

        net.haesleinhuepf.clijx.weka.CLIJxWeka cw2 = new CLIJxWeka(clijx, featureStack, classifier, numberOfClasses);

        ClearCLBuffer buffer = cw2.getClassification();
        clijx.show(buffer, "classification");


        clijx.clear();
    }
}
