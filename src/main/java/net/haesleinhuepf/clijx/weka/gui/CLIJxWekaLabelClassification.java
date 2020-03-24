package net.haesleinhuepf.clijx.weka.gui;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clijx.CLIJx;

import java.awt.*;
import java.util.ArrayList;

public class CLIJxWekaLabelClassification extends CLIJxWekaObjectClassification {


    protected void generateROIs(ImagePlus imp) {
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();

        ClearCLBuffer labelled = clijx.push(imp);

        RoiManager roiManager = new RoiManager(false);
        clijx.pullLabelsToROIManager(labelled, roiManager);

        for (int i = 0; i < roiManager.getCount(); i++) {
            Roi roi = roiManager.getRoi(i);
            roi.setStrokeColor(Color.white);
            //roi.setStrokeWidth(1);
            roi.setName("");
            overlay.add(roi);
        }
        inputImp.setOverlay(overlay);
    }

    protected boolean showInitialDialog() {
        GenericDialogPlus gd = new GenericDialogPlus("CLIJx Weka Label Classification");
        ArrayList<String> deviceNameList = CLIJ.getAvailableDeviceNames();
        String[] deviceNameArray = new String[deviceNameList.size()];
        deviceNameList.toArray(deviceNameArray);
        gd.addChoice("OpenCL Device", deviceNameArray, CLIJxWekaPropertyHolder.clDeviceName);
        gd.addImageChoice("Input image", IJ.getImage().getTitle());
        gd.addImageChoice("Label map", IJ.getImage().getTitle());
        gd.addNumericField("Number of object classes (minimum: 2)", CLIJxWekaPropertyHolder.numberOfObjectClasses, 0);
        gd.showDialog();

        if (gd.wasCanceled()) {
            return false;
        }


        CLIJxWekaPropertyHolder.clDeviceName = gd.getNextChoice();

        inputImp = gd.getNextImage();
        binaryImp = gd.getNextImage();

        CLIJxWekaPropertyHolder.numberOfObjectClasses = (int)gd.getNextNumber();
        return true;
    }



    public static void main(String... args) {
        String imageFilename = "C:/structure/data/2019-12-17-16-54-37-81-Lund_Tribolium_nGFP_TMR/processed/01_max_image/001200.raw.tif";
        String labelsFilename = "C:/structure/data/2019-12-17-16-54-37-81-Lund_Tribolium_nGFP_TMR/processed/07_max_cells/001200.raw.tif";

        new ImageJ();

        ImagePlus imp = IJ.openImage(imageFilename);
        ImagePlus lab = IJ.openImage(labelsFilename);
        IJ.run(imp,"Rotate 90 Degrees Right", "");
        IJ.run(lab,"Rotate 90 Degrees Right", "");

        CLIJx clijx = CLIJx.getInstance();
        ClearCLBuffer input = clijx.push(imp);
        ClearCLBuffer labels = clijx.push(lab);

        clijx.show(input,"input");
        clijx.show(labels, "labels");
        clijx.clear();

        new CLIJxWekaLabelClassification().run(null);
    }

}
