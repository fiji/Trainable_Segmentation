package net.haesleinhuepf.clijx.weka.gui;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.measure.ResultsTable;
import ij.plugin.filter.Analyzer;
import ij.plugin.filter.PlugInFilter;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.weka.ApplyOCLWekaModel;
import net.haesleinhuepf.clijx.weka.CLIJxWeka;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;


public class CLIJxWekaObjectClassification implements PlugInFilter {

    ImagePlus inputImp;
    ImagePlus binaryImp;
    Overlay overlay = new Overlay();
    ResultsTable measurements;

    @Override
    public int setup(String arg, ImagePlus imp) {
        return PlugInFilter.DOES_ALL;
    }

    @Override
    public void run(ImageProcessor ip) {
        inputImp = IJ.getImage();
        binaryImp = IJ.getImage();

        if (!showInitialDialog()) {
            return;
        }

        CLIJxWekaPropertyHolder.setCLIJx(CLIJx.getInstance(CLIJxWekaPropertyHolder.clDeviceName));

        buildGUI();

        ImageWindow window = inputImp.getWindow();
        mouseListener1 = new MouseAdapter() {
            @Override
            public void mouseReleased(MouseEvent e) {
                mouseUp(e);
            }
        };
        window.getCanvas().addMouseListener(mouseListener1);

        System.out.println("setup mouse");
        mouseListener2 = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                System.out.println("mouse " + e.getY());
                int x = e.getXOnScreen();
                int y = e.getYOnScreen();
                if (x > window.getX() && x < window.getX() + window.getWidth() && y > window.getY()) {
                    if (y < window.getY() + window.getCanvas().getY()) {
                        System.out.println("in");
                        guiPanel.setVisible(true);
                        guiPanel.setEnabled(true);
                        refreshGUI();
                        guiPanel.show();
                    } else if (y > window.getY() + window.getCanvas().getY() + guiPanel.getHeight()) {
                        System.out.println("out1");
                        guiPanel.setVisible(false);
                        guiPanel.setEnabled(false);
                    }
                } else {
                    System.out.println("out2");
                }
            }
        };
        window.addMouseMotionListener(mouseListener2);

        mouseListener3 = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                guiPanel.setVisible(false);
                guiPanel.setEnabled(false);
            }
        };
        window.getCanvas().addMouseMotionListener(mouseListener3);


        segmentImage(binaryImp);
    }

    private void segmentImage(ImagePlus imp) {
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();

        ClearCLBuffer binary = clijx.pushCurrentSlice(imp);
        ClearCLBuffer temp = clijx.create(binary);

        ClearCLBuffer labelled = clijx.create(binary.getDimensions(), NativeTypeEnum.Float);

        clijx.binaryFillHoles(binary, temp);
        clijx.connectedComponentsLabelingBox(temp, labelled);

        RoiManager roiManager = new RoiManager(false);
        clijx.pullLabelsToROIManager(labelled, roiManager);

        for (int i = 0; i < roiManager.getCount(); i++) {
            Roi roi = roiManager.getRoi(i);
            roi.setStrokeColor(Color.white);
            roi.setStrokeWidth(2);
            roi.setName("");
            overlay.add(roi);
        }
        inputImp.setOverlay(overlay);

        //easurements = roiManager.(inputImp);
        //measurements.show("Measurements");


    }

    MouseAdapter mouseListener1;
    MouseAdapter mouseListener2;
    MouseAdapter mouseListener3;

    ArrayList<Button> buttons = new ArrayList<>();
    HashMap<Integer, Color> colors = new HashMap<Integer, Color>();

    Dialog guiPanel;
    private void buildGUI() {
        final ImageWindow window = inputImp.getWindow();

        try {
            UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");  // This line gives Windows Theme
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        guiPanel = new Dialog(window);
        //guiPanel.setLayout(new GridLayout(1, 7));
        guiPanel.setLayout(new GridBagLayout());
        guiPanel.setUndecorated(true);

        for (int c = 0; c < CLIJxWekaPropertyHolder.numberOfObjectClasses; c++) {
            final int class_id = c + 1;
            Color color = getColor(class_id);

            Button button = new Button("C" + (c + 1));
            button.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    setCurrentClass(class_id);
                }
            });
            //button.setBackground(color);
            guiPanel.add(button);
            buttons.add(button);
            colors.put(class_id, color);

            Button colorButton = new Button(" ");
            colorButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    Color result = JColorChooser.showDialog(guiPanel, "Color", color);//.getColor();
                    if (result == null) {
                        return;
                    }
                    colorButton.setBackground(result);
                    colors.remove(class_id);
                    colors.put(class_id, result);
                }
            });
            colorButton.setBackground(color);
            guiPanel.add(colorButton);
            guiPanel.add(new Label(" "));
        }

        {
            Button button = new Button("Set measurements");
            button.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    IJ.run("Set Measurements...");
                }
            });
            guiPanel.add(button);
        }
        guiPanel.add(new Label(" "));

        {
            Button button = new Button("Train");
            button.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    trainClicked();
                }
            });
            guiPanel.add(button);
        }
        guiPanel.add(new Label(" "));

        {
            Button button = new Button("Reset");
            button.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    resetClicked();
                }
            });
            guiPanel.add(button);
        }
        guiPanel.add(new Label(" "));

        /*
        int width = 0;
        for (int i = 0; i < guiPanel.getComponentCount(); i++) {
            width = width + guiPanel.getComponent(i).getWidth();
        }*/

        refreshGUI();
    }

    private void resetClicked() {
        for (int i = 0; i < overlay.size(); i++) {
            Roi roi = overlay.get(i);
            roi.setFillColor(null);
        }
        inputImp.updateAndDraw();
    }

    private void trainClicked() {
        ResultsTable table = new ResultsTable();
        //IJ.run("Clear Results");

        Analyzer.setResultsTable(table);
        Analyzer analyser = new Analyzer();


        int measurementsConfig = Analyzer.getMeasurements();

        for (int i = 0; i < overlay.size(); i++) {
            Roi roi = overlay.get(i);
            inputImp.setRoi(roi);
            ImageStatistics stats = inputImp.getStatistics(measurementsConfig);
            analyser.saveResults(stats, roi);
            if (roi.getName() != null && roi.getName().length() > 0) {
                table.addValue("CLASS", Integer.parseInt(roi.getName()));
            } else {
                table.addValue("CLASS", 0);
            }
        }
        inputImp.killRoi();
        Analyzer.setResultsTable(new ResultsTable());

        //table.show("My Results");
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();

        ClearCLBuffer temp = clijx.create(table.getHeadings().length, table.getCounter());
        clijx.resultsTableToImage2D(temp, table);

        ClearCLBuffer transposed1 = clijx.create(temp.getHeight(), temp.getWidth());
        ClearCLBuffer transposed2 = clijx.create(temp.getHeight(), 1, temp.getWidth());
        clijx.transposeXY(temp, transposed1);
        clijx.transposeYZ(transposed1, transposed2);

        ClearCLBuffer ground_truth = clijx.create(transposed2.getWidth(), transposed2.getHeight(), 1);
        ClearCLBuffer featureStack = clijx.create(transposed2.getWidth(), transposed2.getHeight(), transposed2.getDepth() - 1);

        clijx.crop3D(transposed2, featureStack, 0, 0, 0);
        clijx.crop3D(transposed2, ground_truth, 0, 0, transposed2.getDepth() - 1);

        System.out.println("Ground truth:");
        clijx.print(ground_truth);

        CLIJxWeka clijxweka = new CLIJxWeka(clijx, featureStack, ground_truth);
        clijxweka.getClassifier();
        //clijxweka.saveClassifier("temp.model");

        ClearCLBuffer result = clijx.create(ground_truth);

        ApplyOCLWekaModel.applyOCL(clijx, featureStack, result, clijxweka.getOCL());

        float[] resultArray = new float[(int) result.getWidth()];
        FloatBuffer buffer = FloatBuffer.wrap(resultArray);

        result.writeTo(buffer, true);

        for (int i = 0; i < resultArray.length; i++) {
            int class_id = (int)resultArray[i] + 1;
            Color color = getColor(class_id);
            overlay.get(i).setFillColor(new Color(color.getRed(), color.getGreen(), color.getBlue(), 128));
        }
    }

    private void refreshGUI() {
        final ImageWindow window = inputImp.getWindow();
        guiPanel.setSize(window.getCanvas().getWidth() + 2, 30);
        guiPanel.setLocation(window.getX() + window.getCanvas().getX() - 1, window.getY() + window.getCanvas().getY() - 1);

        for (int c = 0; c < buttons.size(); c++) {
            buttons.get(c).setBackground((c + 1 == currentClass)?getColor(c + 1):null);
        }


    }

    private int currentClass = 1;
    private void setCurrentClass(int class_id) {
        for (Button button : buttons) {
            button.setBackground(null);
        }
        currentClass = class_id;
        buttons.get(currentClass - 1).setBackground(getColor(currentClass));
    }

    private Color getColor(int c) {
        switch((c - 1) % 10) {
            case 0:
                return Color.green;
            case 1:
                return Color.magenta;
            case 2:
                return Color.cyan;
            case 3:
                return Color.yellow;
            case 4:
                return Color.red;
            case 5:
                return Color.blue;
            case 6:
                return Color.gray;
            case 7:
                return Color.orange;
            case 8:
                return Color.pink;
            case 9:
                return Color.lightGray;
        }
        return Color.white;
    }


    private void mouseUp(MouseEvent e) {
        System.out.println("mouse up");

        int x = inputImp.getWindow().getCanvas().offScreenX(e.getX());
        int y = inputImp.getWindow().getCanvas().offScreenY(e.getY());

        for (int i = 0; i < overlay.size(); i++) {
            Roi roi = overlay.get(i);
            //System.out.println("Pos: " + x + "/" + y);
            if (roi.contains(x,y)) {
                if (roi.getStrokeColor() == Color.white) {
                    roi.setStrokeColor(colors.get(currentClass));
                    roi.setName("" + currentClass);
                } else {
                    roi.setStrokeColor(Color.white);
                    roi.setName(null);
                }
            }
        }
    }

    private boolean showInitialDialog() {
        GenericDialogPlus gd = new GenericDialogPlus("CLIJx Weka Object Classification");
        ArrayList<String> deviceNameList = CLIJ.getAvailableDeviceNames();
        String[] deviceNameArray = new String[deviceNameList.size()];
        deviceNameList.toArray(deviceNameArray);
        gd.addChoice("OpenCL Device", deviceNameArray, CLIJxWekaPropertyHolder.clDeviceName);
        gd.addImageChoice("Input image", IJ.getImage().getTitle());
        gd.addImageChoice("Binary image (segmented objects)", IJ.getImage().getTitle());
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
        String filename = "C:/structure/data/Kota/NPC_T01.tif";

        new ImageJ();

        ImagePlus imp = IJ.openImage(filename);
        //imp = new Duplicator().run(imp, 2,2,1,1,1,1);
        //imp.show();

        CLIJx clijx = CLIJx.getInstance();
        imp.setC(2);
        ClearCLBuffer input = clijx.pushCurrentSlice(imp);
        clijx.show(input,"input");
        ClearCLBuffer thresholded = clijx.create(input);
        ClearCLBuffer temp = clijx.create(input);
        clijx.gaussianBlur2D(input, temp, 3,3);
        clijx.thresholdOtsu(temp, thresholded);
        clijx.show(thresholded, "thresholded");

        new CLIJxWekaObjectClassification().run(null);
    }


}
