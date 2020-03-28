package net.haesleinhuepf.clijx.weka.autocontext;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.haesleinhuepf.clij2.utilities.HasLicense;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.utilities.AbstractCLIJxPlugin;
import net.haesleinhuepf.clijx.weka.*;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 *         November 2019
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainAutoContextWekaModel")
public class TrainAutoContextWekaModel extends AbstractCLIJxPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, HasAuthor, HasLicense {

    @Override
    public boolean executeCL() {
        ClearCLBuffer input = (ClearCLBuffer)( args[0]);
        ClearCLBuffer ground_truth = (ClearCLBuffer)(args[1]);
        String modelFilename =  (String)args[2];
        String featureDefinitions = (String)args[3];
        int numberOfAutoContextIterations = asInteger(args[4]);
        int numberOfTrees = asInteger(args[5]);
        int numberOfFeatures = asInteger(args[6]);
        int maxDepth = asInteger(args[7]);

        return trainAutoContextWekaModelWithOptions(getCLIJx(), input, ground_truth, modelFilename, featureDefinitions, numberOfAutoContextIterations, numberOfTrees, numberOfFeatures, maxDepth);
    }

    @Override
    public String getParameterHelpText() {
        return "Image input, Image ground_truth, String model_filename, String feature_definitions, Number numberOfAutoContextIterations, Number numberOfTrees, Number numberOfFeatures, Number maxDepth";
    }

    public static boolean trainAutoContextWekaModelWithOptions(CLIJx clijx, ClearCLBuffer input2D, ClearCLBuffer srcGroundTruth2D, String saveModelFilename, String featureDefinitions, int numberOfAutoContextIterations, Integer numberOfTrees, Integer numberOfFeatures, Integer maxDepth) {

        ClearCLBuffer feature_stack = GenerateFeatureStack.generateFeatureStack(clijx, input2D, featureDefinitions);
        int numberOfGeneratedFeatures = (int) feature_stack.getDepth();

        // -------------------------------------------------------------------
        // train classifier, save it as .0.model and as .0.model.cl file
        String model_filename = saveModelFilename + ".0.model";
        CLIJxWeka2 clijxweka = TrainWekaModelWithOptions.trainWekaModelWithOptions(clijx, feature_stack, srcGroundTruth2D, model_filename, numberOfTrees, numberOfFeatures, maxDepth);

        // get probabilities from the first round
        int numberOfClasses = clijxweka.getNumberOfClasses();
        ClearCLBuffer probability_map = clijx.create(new long[]{input2D.getWidth(), input2D.getHeight(), numberOfClasses}, NativeTypeEnum.Float);

        System.out.println("numberOfAutoContextIterations " + numberOfAutoContextIterations);
        for (int iterationCount = 0; iterationCount < numberOfAutoContextIterations; iterationCount++) {
            System.out.println("i" + iterationCount + ": generate probability maps");
            GenerateWekaProbabilityMaps.generateWekaProbabilityMaps(clijx, feature_stack, probability_map, model_filename);
            clijx.show(probability_map,"probability map");

            ClearCLBuffer probability_slice = clijx.create(new long[]{input2D.getWidth(), input2D.getHeight()}, input2D.getNativeType());
            ClearCLBuffer class_feature_stack = clijx.create(new long[]{input2D.getWidth(), input2D.getHeight(), numberOfGeneratedFeatures}, input2D.getNativeType());
            ClearCLBuffer combined_feature_stack = null;
            for (int c = 0; c < numberOfClasses; c++) {
                clijx.copySlice(probability_map, probability_slice, c);

                System.out.println("i" + iterationCount + " c" + c + ": generate feature stack");
                GenerateFeatureStack.generateFeatureStack(clijx, probability_slice, class_feature_stack, featureDefinitions);

                if (combined_feature_stack != null) {
                    clijx.release(combined_feature_stack);
                }
                combined_feature_stack = clijx.create(new long[]{input2D.getWidth(), input2D.getHeight(), feature_stack.getDepth() + class_feature_stack.getDepth()});
                clijx.concatenateStacks(feature_stack, class_feature_stack, combined_feature_stack);

                ClearCLBuffer temp = combined_feature_stack;
                combined_feature_stack = feature_stack;
                feature_stack = temp;
            }
            clijx.release(probability_slice);
            clijx.release(class_feature_stack);
            clijx.release(combined_feature_stack);

            model_filename = saveModelFilename + "." + (iterationCount + 1) + ".model";
            System.out.println("i" + iterationCount + ": train");
            TrainWekaModelWithOptions.trainWekaModelWithOptions(clijx, feature_stack, srcGroundTruth2D, model_filename, numberOfTrees, numberOfFeatures, maxDepth);

        }

        clijx.release(probability_map);
        clijx.release(feature_stack);

        clijx.reportMemory();
        System.out.println("Bye.");

        return true;
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin.\n" +
                "It generates a 3D feature stack as described in GenerateFeatureStack" +
                "and trains a Weka model. This model will be saved to disc.\n" +
                "The given groundTruth image is supposed to be a label map where pixels with value 1 represent class 1, " +
                "pixels with value 2 represent class 2 and so on. Pixels with value 0 will be ignored for training.\n\n" +
                "Default values for options are:\n" +
                "* trees = 200\n" +
                "* features = 2\n" +
                "* maxDepth = 0";
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
        return "Parts of the code CLIJ Weka were copied over from the Trainable_Segmentation repository (link above). Thus,\n" +
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
