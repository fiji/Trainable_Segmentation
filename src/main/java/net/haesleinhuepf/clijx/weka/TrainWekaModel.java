package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.utilities.AbstractCLIJxPlugin;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.haesleinhuepf.clij2.utilities.HasLicense;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 *         November 2019
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainWekaModel")
public class TrainWekaModel extends AbstractCLIJxPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, HasAuthor, HasLicense {

    @Override
    public boolean executeCL() {
        trainWekaModel(getCLIJx(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
        return true;
    }

    public static CLIJxWeka2 trainWekaModel(CLIJx clijx, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer srcGroundTruth2D, String saveModelFilename) {
        CLIJxWeka2 weka = new CLIJxWeka2(clijx, srcFeatureStack3D, srcGroundTruth2D);
        weka.saveClassifier(saveModelFilename);
        return weka;
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image groundTruth2D, String saveModelFilename";
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin.\n" +
                "It takes a 3D feature stack (e.g. first plane original image, second plane blurred, third plane edge image)" +
                "and trains a Weka model. This model will be saved to disc.\n" +
                "The given groundTruth image is supposed to be a label map where pixels with value 1 represent class 1, " +
                "pixels with value 2 represent class 2 and so on. Pixels with value 0 will be ignored for training.";
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
