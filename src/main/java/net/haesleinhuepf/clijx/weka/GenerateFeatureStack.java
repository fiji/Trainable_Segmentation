package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.haesleinhuepf.clij2.utilities.HasLicense;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.utilities.AbstractCLIJxPlugin;
import net.haesleinhuepf.clijx.weka.CLIJxWeka;
import org.scijava.plugin.Plugin;

import java.util.HashMap;

/**
 * Author: @haesleinhuepf
 *         March 2020
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_generateFeatureStack")
public class GenerateFeatureStack extends AbstractCLIJxPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        return generateFeatureStack(getCLIJx(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
    }

    public static boolean generateFeatureStack(CLIJx clijx, ClearCLBuffer input, ClearCLBuffer dstFeatureStack, String featureDefinitions) {
        HashMap<String, ClearCLBuffer> generatedFeatures = new HashMap<String, ClearCLBuffer>();
        String[] definitionsArray = preparseFeatures(featureDefinitions);

        // generate features
        for (String featureDefinition : definitionsArray) {
            generateFeature(clijx, input, generatedFeatures, featureDefinition);
        }

        // collect them in a stack
        int count = 0;
        for (String featureDefinition : definitionsArray) {
            clijx.copySlice(generatedFeatures.get(featureDefinition), dstFeatureStack, count);
            count++;
        }

        // release memory
        for (ClearCLBuffer buffer : generatedFeatures.values()) {
            clijx.release(buffer);
        }

        return true;
    }

    private static ClearCLBuffer generateFeature(CLIJx clijx, ClearCLBuffer input, HashMap<String, ClearCLBuffer> generatedFeatures, String featureDefinition) {
        if (generatedFeatures.containsKey(featureDefinition)) {
            return generatedFeatures.get(featureDefinition);
        }
        String[] temp = featureDefinition.split("=");
        String featureName = temp[0];
        String parameter = temp.length>1?temp[1]:"0";
        double numericParameter = Double.parseDouble(parameter);

        ClearCLBuffer output = clijx.create(input);
        if (featureName.compareTo("original") == 0) {
            clijx.copy(input, output);
        } else if (featureName.compareTo("gaussianblur") == 0) {
            clijx.gaussianBlur2D(input, output, numericParameter, numericParameter);
        } else if (featureName.compareTo("gradientx") == 0) {
            clijx.gradientX(input, output);
        } else if (featureName.compareTo("gradienty") == 0) {
            clijx.gradientY(input, output);
        } else if (featureName.compareTo("minimum") == 0) {
            clijx.minimum2DBox(input, output, numericParameter, numericParameter);
        } else if (featureName.compareTo("maximum") == 0) {
            clijx.maximum2DBox(input, output, numericParameter, numericParameter);
        } else if (featureName.compareTo("mean") == 0) {
            clijx.mean2DBox(input, output, numericParameter, numericParameter);
        } else if (featureName.compareTo("entropy") == 0) {
            clijx.entropyBox(input, output, numericParameter, numericParameter, 0);
        } else if (featureName.compareTo("laplacianofgaussian") == 0) {
            ClearCLBuffer gaussianBlurred = generateFeature(clijx, input, generatedFeatures, "gaussianblur=" + parameter);
            clijx.laplaceBox(gaussianBlurred, output);
        } else if (featureName.compareTo("sobelofgaussian") == 0) {
            ClearCLBuffer gaussianBlurred = generateFeature(clijx, input, generatedFeatures, "gaussianblur=" + parameter);
            clijx.sobel(gaussianBlurred, output);
        } else {
            throw new IllegalArgumentException("Feature " + featureDefinition + "(" + featureName + ") is not supported.");
        }

        generatedFeatures.put(featureDefinition, output);
        return output;
    }

    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        String[] featureDefinitions = preparseFeatures((String)args[2]);

        return getCLIJx().create(new long[]{input.getWidth(), input.getHeight(), featureDefinitions.length}, input.getNativeType());
    }

    private static String[] preparseFeatures(String featureDefinitions) {
        featureDefinitions = featureDefinitions.toLowerCase();
        featureDefinitions = featureDefinitions.trim();
        featureDefinitions = featureDefinitions.replace("\r", " ");
        featureDefinitions = featureDefinitions.replace("\n", " ");
        while (featureDefinitions.contains("  ")) {
            featureDefinitions = featureDefinitions.replace("  ", " ");
        }
        System.out.println("F:" + featureDefinitions);
        return featureDefinitions.split(" ");
    }

    @Override
    public String getParameterHelpText() {
        return "Image input, Image feature_stack_destination, String feature_definitions";
    }

    @Override
    public String getDescription() {
        return "Generates a feature stack for Trainable Weka Segmentation. Use this terminology to specifiy which stacks should be generated:\n" +
                "* \"original\" original slice\n" +
                "* \"GaussianBlur=s\" Gaussian blurred image with sigma s\n" +
                "* \"LaplacianOfGaussian=s\" Laplacian of Gaussian blurred image with sigma s\n" +
                "* \"SobelOfGaussian=s\" Sobel filter applied to Gaussian blurred image with sigma s\n" +
                "* \"minimum=r\" local minimum with radius r\n" +
                "* \"maximum=r\" local maximum with radius r\n" +
                "* \"mean=r\" local mean with radius r\n" +
                "* \"entropy=r\" local entropy with radius r\n" +
                "* \"gradientX\" local gradient in X direction\n" +
                "* \"gradientY\" local gradient in Y direction\n" +
                "\n" +
                "Use sigma=0 to apply a filter to the original image. Feature definitions are not case sensitive.\n" +
                "\n" +
                "Example: \"original gaussianBlur=1 gaussianBlur=5 laplacianOfGaussian=1 laplacianOfGaussian=7 entropy=3\"";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D -> 3D";
    }

}
