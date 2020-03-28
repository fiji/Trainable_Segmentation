package net.haesleinhuepf.clijx.weka;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.util.ElapsedTime;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.haesleinhuepf.clij2.utilities.HasLicense;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.utilities.AbstractCLIJxPlugin;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.HashMap;

/**
 * Author: @haesleinhuepf
 *         November 2019
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_applyOCLWekaModel")
public class ApplyOCLWekaModel extends AbstractCLIJxPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, HasAuthor, HasLicense {

    @Override
    public boolean executeCL() {
        applyOCLWekaModel(getCLIJx(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
        return true;
    }

    public static CLIJxWeka2 applyOCLWekaModel(CLIJx clijx, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer dstClassificationResult, String loadModelFilename) {
        //clijx.stopWatch("");
        CLIJxWeka2 clijxweka = new CLIJxWeka2(clijx, srcFeatureStack3D, loadModelFilename);
        //clijx.stopWatch("init");
        //String ocl = clijxweka.getOCL();
        //applyOCL(clijx, srcFeatureStack3D, dstClassificationResult, ocl);
        ClearCLBuffer classification = clijxweka.getClassification();
        clijx.copy(classification, dstClassificationResult);
        clijx.release(classification);
        return clijxweka;
    }

    public static boolean applyOCL(CLIJx clijx, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer dstClassificationResult, String ocl) {
        //clijx.stopWatch("getocl");
/*
        if (new File(loadModelFilename + ".cl").exists()) {
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("src_featureStack", srcFeatureStack3D);
            parameters.put("dst", dstClassificationResult);
            parameters.put("export_probabilities", 0);
            clijx.execute(Object.class,loadModelFilename + ".cl", "classify_feature_stack", dstClassificationResult.getDimensions(), dstClassificationResult.getDimensions(), parameters);
        } else {
            new IllegalArgumentException("This model hasn't been saved as OCL Model. Try applyWekaModel instead.");
        }
*/
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src_featureStack", srcFeatureStack3D);
        parameters.put("dst", dstClassificationResult);
        parameters.put("export_probabilities", 0);
        //CLIJ.debug = true;
        //ElapsedTime.sStandardOutput = true;
        clijx.executeCode(ocl, "classify_feature_stack", dstClassificationResult.getDimensions(), dstClassificationResult.getDimensions(), parameters);
        //CLIJ.debug = false;
        return true;
    }

    public static void main(String... args) {
        String imageFilename = "C:\\structure\\data\\unidesigner_groundtruth-drosophila-vnc\\stack1\\raw\\00.tif";
        String groundTruthFilename = "C:\\structure\\data\\unidesigner_groundtruth-drosophila-vnc\\stack1\\labels\\labels00000000.png";
        String feature_definition = "original gaussianblur=1 gaussianblur=5 gaussianblur=7 sobelofgaussian=1 sobelofgaussian=5 sobelofgaussian=7";
        String modelFilename = "C:\\structure\\code\\clij_weka_scripts\\em_segm\\test.model";

        new ImageJ();
        ImagePlus imp = IJ.openImage(imageFilename);
        IJ.run(imp, "32-bit", "");
        ImagePlus ground_truth = IJ.openImage(groundTruthFilename);
        IJ.run(ground_truth, "32-bit", "");

        CLIJx clijx = CLIJx.getInstance();
        ClearCLBuffer input = clijx.push(imp);
        ClearCLBuffer input_ground_truth = clijx.push(ground_truth);

        ClearCLBuffer output = clijx.create(input);

        clijx.stopWatch("");
        ClearCLBuffer featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, feature_definition);
        clijx.stopWatch("generate feature stack");

        input_ground_truth = replaceIntensities(clijx, input_ground_truth);
        input_ground_truth = thinOutGroundTruth(clijx, input_ground_truth, 0.99f);
        //input_ground_truth = convertToFloat(clijx, input_ground_truth);

        clijx.show(input_ground_truth, "in gt");

        clijx.stopWatch("thin ground truth");

        TrainWekaModelWithOptions.trainWekaModelWithOptions(clijx, featureStack, input_ground_truth, modelFilename, 200, 2, 5);
        clijx.stopWatch("train");

        //CLIJxWeka weka = new CLIJxWeka(clijx, featureStack, modelFilename);

        ApplyWekaModel.applyWekaModel(clijx, featureStack, output, modelFilename);
        clijx.stopWatch("CPU predict1");

        ApplyWekaModel.applyWekaModel(clijx, featureStack, output, modelFilename);
        clijx.stopWatch("GPU predict2");

        ApplyOCLWekaModel.applyOCLWekaModel(clijx, featureStack, output, modelFilename);
        clijx.stopWatch("GPU predict1");

        ApplyOCLWekaModel.applyOCLWekaModel(clijx, featureStack, output, modelFilename);
        clijx.stopWatch("GPU predict2");

        clijx.show(output, "output");



    }


    private static ClearCLBuffer replaceIntensities(CLIJx clijx, ClearCLBuffer input_ground_truth) {
        ClearCLBuffer temp1 = input_ground_truth;
        ClearCLBuffer temp2 = clijx.create(input_ground_truth);

        int membrane = 1;
        int glia_extracellular = 2;
        int mitochondria = 3;
        int synapse = 4;
        int intracellular = 5;

        // 0   -> membrane | (0°)
        clijx.replaceIntensity(temp1, temp2, 0, membrane);
        // 32  -> membrane / (45°)
        clijx.replaceIntensity(temp2, temp1, 32, membrane);
        // 64  -> membrane - (90°)
        clijx.replaceIntensity(temp1, temp2, 64, membrane);
        // 96  -> membrane \ (135°)
        clijx.replaceIntensity(temp2, temp1, 96, membrane);
        // 128 -> membrane "junction"
        clijx.replaceIntensity(temp1, temp2, 128, membrane);
        // 159 -> glia/extracellular
        clijx.replaceIntensity(temp2, temp1, 159, glia_extracellular);
        // 191 -> mitochondria
        clijx.replaceIntensity(temp1, temp2, 191, mitochondria);
        // 223 -> synapse
        clijx.replaceIntensity(temp2, temp1, 223, synapse);
        // 255 -> intracellular
        clijx.replaceIntensity(temp1, temp2, 255, intracellular);

        clijx.copy(temp2, temp1);

        return input_ground_truth;
    }

    private static ClearCLBuffer thinOutGroundTruth(CLIJx clijx, ClearCLBuffer input_ground_truth, float amount) {
        ClearCLBuffer temp = clijx.create(input_ground_truth);
        ClearCLBuffer temp2 = clijx.create(input_ground_truth);

        clijx.setRandom(temp2, 0f, 1f);
        clijx.greaterConstant(temp2, temp, amount);
        clijx.mask(input_ground_truth, temp, temp2);
        clijx.copy(temp2, input_ground_truth);

        clijx.release(temp);
        clijx.release(temp2);

        return input_ground_truth;
    }


    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        return getCLIJx().create(new long[]{input.getWidth(), input.getHeight()}, NativeTypeEnum.Float);
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image prediction2D_destination, String loadModelFilename";
    }

    @Override
    public String getDescription() {
        return "Applies a Weka model which was saved as OpenCL file. Train your model with trainWekaModel to save it as OpenCL file.\n" +
                "It takes a 3D feature stack (e.g. first plane original image, second plane blurred, third plane edge image)" +
                "and applies a pre-trained a Weka model. Take care that the feature stack has been generated in the same" +
                "way as for training the model!";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }

    @Override
    public String getAuthorName() {
        return "Robert Haase (rhaase@mpi-cbg.de, based on work by \n" +
                " Verena Kaynig (verena.kaynig@inf.ethz.ch), \n" +
                " Ignacio Arganda-Carreras (iarganda@mit.edu),\n" +
                " Albert Cardona (acardona@ini.phys.ethz.ch))";
    }

    @Override
    public String getLicense() {
        return "Parts of the code of CLIJ Weka were copied over from the Trainable_Segmentation repository (link above). Thus,\n" +
                " this code is licensed GPL2 as well.\n" +
                "\n" +
                "  License: GPL\n" +
                "\n" +
                "  This program is free software; you can redistribute it and/or\n" +
                "  modify it under the terms of the GNU General Public License 2\n" +
                "  as published by the Free Software Foundation.\n" +
                "\n" +
                "  This program is distributed in the hope that it will be useful,\n" +
                "  but WITHOUT ANY WARRANTY; without even the implied warranty of\n" +
                "  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" +
                "  GNU General Public License for more details.\n" +
                "\n" +
                "  You should have received a copy of the GNU General Public License\n" +
                "  along with this program; if not, write to the Free Software\n" +
                "  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.\n" +
                "  Authors: Verena Kaynig (verena.kaynig@inf.ethz.ch), Ignacio Arganda-Carreras (iarganda@mit.edu)\n" +
                "           Albert Cardona (acardona@ini.phys.ethz.ch)";
    }
}
