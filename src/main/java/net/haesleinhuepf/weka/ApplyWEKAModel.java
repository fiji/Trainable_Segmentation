package net.haesleinhuepf.weka;

import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.kernels.Kernels;
import net.haesleinhuepf.clij.macro.AbstractCLIJPlugin;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import org.scijava.plugin.Plugin;
import trainableSegmentation.WekaSegmentation;
import trainableSegmentation.Weka_Segmentation;

import java.util.HashMap;

import static net.haesleinhuepf.clij.utilities.CLIJUtilities.assertDifferent;

/**
 * Author: @haesleinhuepf
 *         November 2019
 */

@Plugin(type = CLIJMacroPlugin.class, name = "WEKA_applyWEKAModel")
public class ApplyWEKAModel extends AbstractCLIJPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        return applyWEKAModel(clij, (ClearCLBuffer) args[0], (ClearCLBuffer) args[1], args[2].toString());
    }

    public static boolean applyWEKAModel(CLIJ clij, ClearCLBuffer src, ClearCLBuffer dst, String modelFilename) {
        assertDifferent(src, dst);

        ImagePlus imp = clij.pull(src);

        WekaSegmentation ws = new WekaSegmentation();
        ws.loadClassifier(modelFilename);
        ImagePlus result = ws.applyClassifier(imp);
        ClearCLBuffer resultBuffer = clij.push(result);

        clij.op().copy(resultBuffer, dst);
        result.close();

        return true;
    }

    @Override
    public String getParameterHelpText() {
        return "Image source, Image destination, String model_filename";
    }

    @Override
    public String getDescription() {
        return "Applies a WEKA model to an image.";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }
}
