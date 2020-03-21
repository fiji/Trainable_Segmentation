package net.haesleinhuepf.clijx.weka.gui;

import net.haesleinhuepf.clijx.CLIJx;

class CLIJxWekaPropertyHolder {
    static String pixelClassificationModelFile = "new.model";
    static int numberOfObjectClasses = 2;
    static String clDeviceName = "";
    static String pixelClassificationFeatureDefinition = "original gaussianblur=1 gaussianblur=5 sobelofgaussian=1 sobelofgaussian=5";

    private static CLIJx clijx = null;
    public static CLIJx getCLIJx() {
        if (clijx == null) {
            clijx = CLIJx.getInstance();
        }
        return clijx;
    }
    public static void setCLIJx(CLIJx clijx) {
        CLIJxWekaPropertyHolder.clijx = clijx;
    }
}
