package net.haesleinhuepf.clijx.weka.gui.kernels;

import net.haesleinhuepf.clij.clearcl.interfaces.ClearCLImageInterface;
import net.haesleinhuepf.clijx.CLIJx;

import java.awt.*;
import java.util.HashMap;

/**
 * Author: @haesleinhuepf
 *         March 2020
 */
public class MakeRGB {
    public static boolean makeRGB(CLIJx clijx,
                                  ClearCLImageInterface srcInput,
                                  ClearCLImageInterface srcForeground,
                                  ClearCLImageInterface srcBackground,
                                  ClearCLImageInterface dst,
                                  float min, float max,
                                  Color foreground,
                                  Color background,
                                  float alpha) {

        HashMap<String, Object> parameters = new HashMap<>();


        parameters.put("srcInput", srcInput);
        parameters.put("srcForeground", srcForeground);
        parameters.put("srcBackground", srcBackground);
        parameters.put("dst", dst);

        parameters.put("input_min", min);
        parameters.put("input_max", max);

        parameters.put("input_r", 1f - alpha);
        parameters.put("input_g", 1f - alpha);
        parameters.put("input_b", 1f - alpha);

        parameters.put("foreground_r", new Float(((float)foreground.getRed()) / 255.0 * alpha ));
        parameters.put("foreground_g", new Float(((float)foreground.getGreen()) / 255.0 * alpha ));
        parameters.put("foreground_b", new Float(((float)foreground.getBlue()) / 2550 * alpha ));

        parameters.put("background_r", new Float(((float)background.getRed()) / 255.0 * alpha ));
        parameters.put("background_g",new Float(((float)background.getGreen()) / 255.0 * alpha ));
        parameters.put("background_b",new Float(((float)background.getBlue()) / 255.0 * alpha ));

        clijx.execute(MakeRGB.class, "makeRGB_x.cl", "makeRGB", srcInput.getDimensions(), srcInput.getDimensions(), parameters);
        return true;
    }
}
