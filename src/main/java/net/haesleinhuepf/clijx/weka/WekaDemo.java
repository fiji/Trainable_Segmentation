package net.haesleinhuepf.clijx.weka;

import hr.irb.fastRandomForest_clij.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clijx.CLIJx;
import net.imagej.ops.OpEnvironment;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.roi.labeling.LabelingType;
import net.imglib2.trainable_segmention.classification.CompositeInstance;
import net.imglib2.trainable_segmention.classification.Segmenter;
import net.imglib2.trainable_segmention.classification.Trainer;
import net.imglib2.trainable_segmention.clij_random_forest.*;
import net.imglib2.trainable_segmention.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmention.pixel_feature.filter.SingleFeatures;
import net.imglib2.trainable_segmention.pixel_feature.settings.ChannelSetting;
import net.imglib2.trainable_segmention.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmention.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.StopWatch;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import net.imglib2.view.composite.CompositeIntervalView;
import net.imglib2.view.composite.RealComposite;
import preview.net.imglib2.loops.LoopBuilder;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;

import java.util.*;

public class WekaDemo {
    public static void main(String[] args) {
        long time;
        HashMap<String, Long> durations = new HashMap<>();

        net.imagej.ImageJ ij = new net.imagej.ImageJ();
        ij.ui().showUI();

        // init GPU
        CLIJx clijx = CLIJx.getInstance();

        ClearCLBuffer input = null;
        ClearCLBuffer partialGroundTruth = null;
        ClearCLBuffer featureStack = null;

        // Benchmark CLIJxWeka
        {
            // Load Example data
            ImagePlus inputImp = IJ.openImage("src/test/resources/NPC_T01_c2.tif");
            IJ.run(inputImp, "32-bit", "");
            ImagePlus partialGroundTruthImp = IJ.openImage("src/test/resources/NPC_T01_c2_ground_truth.tif");

            String featureDefinition = "original gaussianblur=1 gaussianblur=5 sobelofgaussian=1 sobelofgaussian=5";

            // push to GPU
            input = clijx.push(inputImp);
            partialGroundTruth = clijx.push(partialGroundTruthImp);

            // -------------------------------------------------------------------------------------------------------------
            // generate feature stack
            time = System.currentTimeMillis();
            featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, featureDefinition);
            clijx.release(featureStack);
            durations.put("A feature stack generation CLIJ 1", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // generate feature stack again
            time = System.currentTimeMillis();
            featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, featureDefinition);
            durations.put("A feature stack generation CLIJ 2", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // show input data
            //new ImageJ();
            clijx.show(input, "input");
            clijx.show(partialGroundTruth, "partial ground truth");
            clijx.show(featureStack, "feature stack");

            // -------------------------------------------------------------------------------------------------------------
            // Train (internally with Fijis Trainable Segmentation)
            time = System.currentTimeMillis();
            CLIJxWeka cw = new CLIJxWeka(clijx, featureStack, partialGroundTruth);

            FastRandomForest classifier = cw.getClassifier();
            int numberOfClasses = cw.getNumberOfClasses();
            durations.put("B Train FastRandomForest using Weka 1", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // Predict (internally with Fijis Trainable Segmentation)
            time = System.currentTimeMillis();
            CLIJxWeka cw2 = new CLIJxWeka(clijx, featureStack, classifier, numberOfClasses);

            ClearCLBuffer buffer = cw2.getClassification();
            durations.put("C Predict FastRandomForest using Weka 1", System.currentTimeMillis() - time);

            clijx.show(buffer, "classification");

            // -------------------------------------------------------------------------------------------------------------
            // Predict (internally with OpenCL)
            time = System.currentTimeMillis();
            String openCLkernelcode = cw2.getOCL();
            ApplyOCLWekaModel.applyOCL(clijx, featureStack, buffer, openCLkernelcode);
            durations.put("C Predict FastRandomForest using CLIJ 1", System.currentTimeMillis() - time);

            clijx.show(buffer, "classification_opencl1");

            // -------------------------------------------------------------------------------------------------------------
            // Predict (internally with OpenCL) again
            time = System.currentTimeMillis();
            ApplyOCLWekaModel.applyOCL(clijx, featureStack, buffer, openCLkernelcode);
            durations.put("C Predict FastRandomForest using CLIJ 2", System.currentTimeMillis() - time);

            clijx.show(buffer, "classification_opencl2");
        }

        // -------------------------------------------------------------------------------------------------------------
        // Benchmark Weka + Labikit kernel
        {
            // Load Example data
            ImagePlus inputImp = IJ.openImage("src/test/resources/NPC_T01_c2.tif");
            IJ.run(inputImp, "32-bit", "");
            ImagePlus partialGroundTruthImp = IJ.openImage("src/test/resources/NPC_T01_c2_ground_truth.tif");

            String featureDefinition = "original gaussianblur=1 gaussianblur=5 sobelofgaussian=1 sobelofgaussian=5";

            // push to GPU
            input = clijx.push(inputImp);
            partialGroundTruth = clijx.push(partialGroundTruthImp);

            // -------------------------------------------------------------------------------------------------------------
            // generate feature stack
            time = System.currentTimeMillis();
            featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, featureDefinition);
            clijx.release(featureStack);
            durations.put("A feature stack generation CLIJ 3", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // generate feature stack again
            time = System.currentTimeMillis();
            featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, featureDefinition);
            durations.put("A feature stack generation CLIJ 4", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // show input data
            //new ImageJ();
            clijx.show(input, "input");
            clijx.show(partialGroundTruth, "partial ground truth");
            clijx.show(featureStack, "feature stack");

            // -------------------------------------------------------------------------------------------------------------
            // Train (internally with Fijis Trainable Segmentation)
            time = System.currentTimeMillis();
            CLIJxWeka2 cw = new CLIJxWeka2(clijx, featureStack, partialGroundTruth);

            hr.irb.fastRandomForest.FastRandomForest classifier = cw.getClassifier();
            int numberOfClasses = cw.getNumberOfClasses();
            durations.put("B Train FastRandomForest using Weka 2", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // Predict (internally with Labkit Kernel)
            time = System.currentTimeMillis();
            CLIJxWeka2 cw2 = new CLIJxWeka2(clijx, featureStack, classifier, numberOfClasses);

            ClearCLBuffer buffer = cw2.getClassification();
            durations.put("C Predict FastRandomForest using Weka + Labkit Kernel 1", System.currentTimeMillis() - time);

            clijx.show(buffer, "classification LK1");

            // -------------------------------------------------------------------------------------------------------------
            // Predict (internally with Labkit Kernel)
            time = System.currentTimeMillis();
            CLIJxWeka2 cw3 = new CLIJxWeka2(clijx, featureStack, classifier, numberOfClasses);

            buffer = cw3.getClassification();
            durations.put("C Predict FastRandomForest using Weka + Labkit Kernel 2", System.currentTimeMillis() - time);

            clijx.show(buffer, "classification LK2");

        }

        // -------------------------------------------------------------------------------------------------------------
        //

        // Benchmark imglib2-trainable-segmentation
        if (false)
        {
            // How can I determine these numbers?
            int numberOfFeatures = 10;
            int numberOfClasses = 2;

            // There are errors when working with 2D images, thus we extend them to 3D
            // Question: Is Labkit supposed to work with 2D data?
			// Answer: No 2D is supposed to work as well, the clij branch is still WIP. I need to fix this.
            ClearCLBuffer input3D = clijx.create(new long[]{input.getWidth(), input.getHeight(), 1}, input.getNativeType());
            clijx.copySlice(input, input3D, 0);
            ClearCLBuffer partialGroundTruth3D = clijx.create(new long[]{partialGroundTruth.getWidth(), partialGroundTruth.getHeight(), 1}, partialGroundTruth.getNativeType());
            clijx.copySlice(partialGroundTruth, partialGroundTruth3D, 0);
            // ClearCLBuffer input3D = input;
            // ClearCLBuffer partialGroundTruth3D = ClearCLBuffer partialGroundTruth;

            RandomAccessibleInterval inputRAI = clijx.pullRAI(input3D);
            RandomAccessibleInterval featureStackRAI = clijx.pullRAI(featureStack); //unused; would be cool though
            RandomAccessibleInterval partialGroundTruthRAI = clijx.pullRAI(partialGroundTruth3D);

            // There are errors when working with 2D images, thus we extend them to 3D
            inputRAI = Views.addDimension(inputRAI, 0, 0);
            partialGroundTruthRAI = Views.addDimension(partialGroundTruthRAI, 0, 0);

            OpEnvironment ops = ij.op();
            RandomAccessibleInterval labelingRai = clijx.pullRAI(partialGroundTruthRAI);
            LabelRegions labeling = raiToLabeling(ops, labelingRai);

            List<String> classNames = new ArrayList<>();
            classNames.add("1");
            classNames.add("2");

            // Question: Is it possible to hand over a feature-stack-RAI? If yes, with which dimensionality/shape?
			// Yes, see below
			final FeatureSettings featureSettings = new FeatureSettings(GlobalSettings.default3d().build(),
                    // Question: How can I enter the original image as feature? Or is it there by default? How could I remove it?
					// Answer: use SingleFeatures.identity()
                    GroupedFeatures.gauss(),
					// Question: How can I enter custom radii?
					// There are two ways to do so:
					// Answer 1, specify custom radii in the global settings: GlobalSettings.default2d().sigmas(1, 3, 9).build()
					// Answer 2, use SingleFeautres: SingleFeatures.gauss(2.5), SingleFeatures.gauss(7.7)
                    //GroupedFeatures.differenceOfGaussians(),
                    //GroupedFeatures.hessian(),
                    GroupedFeatures.gradient()); // Question: I would like to use the gradient of different Gaussians as feature, how can I specify this?
					// Answer all the GroupedFeatures are applied to all the sigmas specified with GlobalSettings


            // Question: In CLIJxWeka, I struggle saving feature-definition and classifier together in a file.
            //           However, that's kind of a must. How would you do this? What format shall we use for this?
			// I write it all together in one json file, that's what {@link Segmenter#toJson()} is for.
			// We should definitely talk about this.

            // -------------------------------------------------------------------------------------------------------------
            // train using imglib2-trainable-segmentsion
            time = System.currentTimeMillis();
            Segmenter segmenter = Trainer.train(ops, inputRAI, labeling, featureSettings);
            durations.put("D Train FastRandomForest using imglib2-trainable-segmentation", System.currentTimeMillis() - time);
            AbstractClassifier initRandomForest = (AbstractClassifier) segmenter.getClassifier();

            // -------------------------------------------------------------------------------------------------------------
            // predict using imglib2-trainable-segmentation
            time = System.currentTimeMillis();
            // Question: Is there a way to create a Segmenter from a classifier?
			// No, It would be very hard to train it with the correct set of features.
			// Why would you want to do that?
            //Segmenter segmenter = new Segmenter(ops, classNames, featureSettings, initRandomForest);
            RandomAccessibleInterval result = segmenter.segment(inputRAI);
            durations.put("E Predict FastRandomForest using imglib2-trainable-segmentation", System.currentTimeMillis() - time);
            ImageJFunctions.show(result);

            // -------------------------------------------------------------------------------------------------------------
            // generate features using imglib2-trainable-segmentation-CLIJ
            time = System.currentTimeMillis();
            CLIJMultiChannelImage featuresCl = calculateFeatures(clijx, input, numberOfFeatures);
            durations.put("E generate features using imglib2-trainable-segmentation-CLIJ 1", System.currentTimeMillis() - time);
            clijx.show(featuresCl.asClearCLBuffer(), "featuresCL");

            // -------------------------------------------------------------------------------------------------------------
            // generate features using imglib2-trainable-segmentation-CLIJ again
            time = System.currentTimeMillis();
            featuresCl = calculateFeatures(clijx, input3D, numberOfFeatures);
            durations.put("E generate features using imglib2-trainable-segmentation-CLIJ 2", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // predict using imglib2-trainable-segmentation-CLIJ
            time = System.currentTimeMillis();
            CLIJMultiChannelImage distributionCl = calculateDistribution(clijx, initRandomForest, featuresCl, numberOfClasses, numberOfFeatures);
            ClearCLBuffer segmentationCl = calculateSegmentation(clijx, distributionCl);
            durations.put("F predict using imglib2-trainable-segmentation-CLIJ 1", System.currentTimeMillis() - time);

            // -------------------------------------------------------------------------------------------------------------
            // predict using imglib2-trainable-segmentation-CLIJ again
            time = System.currentTimeMillis();
            distributionCl = calculateDistribution(clijx, initRandomForest, featuresCl, numberOfClasses, numberOfFeatures);
            segmentationCl = calculateSegmentation(clijx, distributionCl);
            durations.put("F predict using imglib2-trainable-segmentation-CLIJ 2", System.currentTimeMillis() - time);

            clijx.show(segmentationCl, "segmentation imglib2-t-s-clij");

        }

        // -------------------------------------------------------------------------------------------------------------
        // output and cleanup memory
        System.out.println(clijx.reportMemory());
        clijx.clear();

        // -------------------------------------------------------------------------------------------------------------
        // output durations
        ArrayList<String> list = new ArrayList<>();
        list.addAll(durations.keySet());
        Collections.sort(list);
        for (String key : list) {
            System.out.println(key + ": " + durations.get(key));
        }

    }

	// Question: Is it possible to hand over a feature-stack-RAI? If yes, with which dimensionality/shape?
	// Yes, it's possible, channel order should be XYC or XYZC
	private RandomAccessibleInterval< UnsignedByteType > Answer(RandomAccessibleInterval< FloatType > featureStack,
			LabelRegions< ? > labeling, OpEnvironment ops) {
    	int numberOfChannels = (int) featureStack.dimension(2);
		GlobalSettings globalSettings = GlobalSettings.default2d().channels(ChannelSetting.multiple(numberOfChannels)).build();
		FeatureSettings featureSettings = new FeatureSettings(globalSettings, SingleFeatures.identity());
		Segmenter segmenter = Trainer.train(ops, featureStack, labeling, featureSettings);
		return segmenter.segment(featureStack);
	}

    // Question: Is there an easier way for doing this?
    // source: https://forum.image.sc/t/construct-labelregions-from-labelmap/20590/2
	// Answer: There is a new method ImgLabling.fromImageAndLabels(...) which should make think's a little bit simpler.
    private static LabelRegions raiToLabeling(OpEnvironment ops, RandomAccessibleInterval labelingRai) {

        RandomAccessibleInterval<IntType> img = ops.convert().int32(Views.iterable(labelingRai));


        final Dimensions dims = img;
        final IntType t = new IntType();
        final RandomAccessibleInterval<IntType> labelImg = Util.getArrayOrCellImgFactory(dims, t).create(dims, t);
        ImgLabeling<Integer, IntType> labelingImg = new ImgLabeling<Integer, IntType>(labelImg);

        // create labeling image
        final Cursor<LabelingType<Integer>> labelCursor = Views.flatIterable(labelingImg).cursor();

        for (final IntType input : Views.flatIterable(img)) {
            final LabelingType<Integer> element = labelCursor.next();
            if (input.getRealFloat() != 0) {
                element.add((int) input.getRealFloat());
            }
        }
        return new LabelRegions(labelingImg);
    }


    // following methods were copied / adapted from
    // https://github.com/maarzt/imglib2-trainable-segmentation/blob/clij/src/test/java/clij/CLIJDemo.java#L76-L125
    private static ClearCLBuffer calculateSegmentation(CLIJ2 clij, CLIJMultiChannelImage distribution) {
        ClearCLBuffer result = clij.create(distribution.getSpatialDimensions(), NativeTypeEnum.Float);
        CLIJRandomForestKernel.findMax(clij, distribution, result);
        return result;
    }

    private static CLIJMultiChannelImage calculateDistribution(CLIJ2 clij, Classifier classifier,
                                                               CLIJMultiChannelImage featuresCl, int numberOfClasses, int numberOfFeatures)
    {
        RandomForestPrediction prediction = new RandomForestPrediction((hr.irb.fastRandomForest.FastRandomForest) classifier, numberOfClasses, numberOfFeatures);
        CLIJMultiChannelImage output = new CLIJMultiChannelImage(clij, featuresCl.getSpatialDimensions(), numberOfClasses);
        prediction.distribution(clij, featuresCl, output);
        return output;
    }

    private static Img<UnsignedByteType> segment(Classifier classifier, Attribute[] attributes,
                                                 ImagePlus output, int numberOfClasses, int numberOfFeatures)
    {
        RandomForestPrediction prediction = new RandomForestPrediction((hr.irb.fastRandomForest.FastRandomForest) classifier,
                numberOfClasses, numberOfFeatures);
        RandomAccessibleInterval<FloatType> featureStack =
                Views.permute(ImageJFunctions.wrapFloat(output), 2, 3);
        CompositeIntervalView<FloatType, RealComposite<FloatType>> collapsed =
                Views.collapseReal(featureStack);
        CompositeInstance compositeInstance =
                new CompositeInstance(collapsed.randomAccess().get(), attributes);
        Img<UnsignedByteType> segmentation =
                ArrayImgs.unsignedBytes(Intervals.dimensionsAsLongArray(collapsed));
        StopWatch stopWatch = StopWatch.createAndStart();
        LoopBuilder.setImages(collapsed, segmentation).forEachPixel((c, o) -> {
            compositeInstance.setSource(c);
            o.set(prediction.classifyInstance(compositeInstance));
        });
        System.out.println(stopWatch);
        return segmentation;
    }

    private static CLIJMultiChannelImage calculateFeatures(CLIJ2 clij, ClearCLBuffer inputCl, int numberOfFeatures) {
        try (ClearCLBuffer tmpCl = clij.create(inputCl)) {
            CLIJMultiChannelImage output = new CLIJMultiChannelImage(clij, inputCl.getDimensions(), numberOfFeatures);
            List<CLIJView> slices = output.channels();
            for (int i = 0; i < numberOfFeatures; i++) {
                float sigma = i * 2;
                clij.gaussianBlur(inputCl, tmpCl, sigma, sigma, sigma);
                CLIJCopy.copy(clij, CLIJView.wrap(tmpCl), slices.get(i));
            }
            return output;
        }
    }
}
