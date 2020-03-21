package net.haesleinhuepf.clijx.weka.gui;

import fiji.util.gui.GenericDialogPlus;
import ij.*;
import ij.gui.*;
import ij.io.FileSaver;
import ij.io.SaveDialog;
import ij.plugin.HyperStackConverter;
import ij.plugin.RGBStackConverter;
import ij.plugin.Selection;
import ij.plugin.filter.PlugInFilter;
import ij.plugin.frame.Recorder;
import ij.process.FloatPolygon;
import ij.process.ImageProcessor;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clijx.CLIJx;
import net.haesleinhuepf.clijx.weka.ApplyOCLWekaModel;
import net.haesleinhuepf.clijx.weka.CLIJxWeka;
import net.haesleinhuepf.clijx.weka.GenerateFeatureStack;
import net.haesleinhuepf.clijx.weka.TrainWekaModel;
import net.haesleinhuepf.clijx.weka.gui.kernels.MakeRGB;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;

public class CLIJxWekaBinaryPixelClassification  implements PlugInFilter {

    Color defaultRoiColor;

    static Color foregroundColor = Color.green;
    static Color backgroundColor = Color.magenta;
    static Color eraseColor = Color.yellow;
    float overlayAlpha = 0.5f;


    ImagePlus inputImp = null;

    Overlay overlay;
    TextRoi status;

    ClearCLBuffer clInput = null;
    ClearCLBuffer clResult = null;
    CLIJxWeka clijxweka = null;

    public CLIJxWekaBinaryPixelClassification() {
        overlay = new Overlay();
        status = new TextRoi("o", 0, 15, new Font("Arial", Font.PLAIN, 15));
    }


    @Override
    public int setup(String arg, ImagePlus imp) {
        return PlugInFilter.DOES_ALL;
    }

    GenericDialogPlus gdp;

    @Override
    public void run(ImageProcessor ip) {
        inputImp = IJ.getImage();
        defaultRoiColor = Roi.getColor();

        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();

        clInput = clijx.push(inputImp);


        buildGUI();
        updateVisualisation();

        addForegroundClicked();

        imageListener = new ImageListener() {
            @Override
            public void imageOpened(ImagePlus imp) {

            }

            @Override
            public void imageClosed(ImagePlus imp) {

            }

            @Override
            public void imageUpdated(ImagePlus imp) {
                if (imp == inputImp) {
                    imageChanged();
                }
            }
        };
        ImagePlus.addImageListener(imageListener);


        mouseListener1 = new MouseAdapter() {
            @Override
            public void mouseReleased(MouseEvent e) {
                mouseUp(e);
            }
        };
        inputImp.getWindow().getCanvas().addMouseListener(mouseListener1);
    }

    ImageListener imageListener;
    MouseAdapter mouseListener1;
    MouseAdapter mouseListener2;
    MouseAdapter mouseListener3;


    Dialog guiPanel;
    JToggleButton foregroundButton;
    JToggleButton backgroundButton;
    JToggleButton eraseButton;
    Button saveButton;
    Button applyButton;

    Button foregroundColorButton;
    Button backgroundColorButton;
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



        // window.getCanvas().setLocation(0, 100);
        System.out.println("setup mouse");
        mouseListener2 = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                //System.out.println("mouse " + e.getY());
                int x = e.getXOnScreen();
                int y = e.getYOnScreen();
                if (x > window.getX() && x < window.getX() + window.getWidth() && y > window.getY()) {
                    if (y < window.getY() + window.getCanvas().getY()) {
                        //System.out.println("in");
                        guiPanel.setVisible(true);
                        guiPanel.setEnabled(true);
                        refreshGUI();
                        guiPanel.show();
                    } else if (y > window.getY() + window.getCanvas().getY() + guiPanel.getHeight()) {
                        guiPanel.setVisible(false);
                        guiPanel.setEnabled(false);
                    }
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

        /*window.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                refreshGUI();
            }
        });*/

        {
            foregroundButton = new JToggleButton("Foreground");
            foregroundButton.setSize(40, 15);
            foregroundButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    addForegroundClicked();
                }
            });
            guiPanel.add(foregroundButton);
        }
        {
            foregroundColorButton = new Button(" ");
            foregroundColorButton.setSize(15, 15);
            foregroundColorButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    Color result = JColorChooser.showDialog(guiPanel, "Foreground", foregroundColor);//.getColor();
                    if (result == null) {
                        return;
                    }
                    foregroundColor = result;
                    refreshGUI();
                }
            });
            guiPanel.add(foregroundColorButton);
        }
        guiPanel.add(new Label(" "));
        {
            backgroundButton = new JToggleButton("Background");
            backgroundButton.setSize(40, 15);
            backgroundButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    addBackgroundClicked();
                }
            });
            guiPanel.add(backgroundButton);
        }{
            backgroundColorButton = new Button(" ");
            backgroundColorButton.setSize(15, 15);
            backgroundColorButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    Color result = JColorChooser.showDialog(guiPanel, "Background", backgroundColor);//.getColor();
                    if (result == null) {
                        return;
                    }
                    backgroundColor = result;
                    refreshGUI();
                }
            });
            guiPanel.add(backgroundColorButton);
        }
        guiPanel.add(new Label(" "));
        {

            eraseButton = new JToggleButton("Erase");
            eraseButton.setSize(40, 15);
            eraseButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    eraseClicked();
                }
            });
            guiPanel.add(eraseButton);
        }
        guiPanel.add(new Label(" "));

        {
            Button trainButton = new Button("Train");
            trainButton.setSize(40, 15);
            trainButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    trainClicked();
                }
            });
            guiPanel.add(trainButton);
        }

        {
            JSlider slider = new JSlider(0, 100);
            slider.setSize(25, 15);
            slider.setValue((int) (overlayAlpha * 100));
            slider.setToolTipText("Result overlay alpha");
            slider.addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent e) {
                    System.out.println("Changed");
                    overlayAlpha = (float) slider.getValue() / 100;

                    System.out.println("Changed " + overlayAlpha);
                    updateVisualisation();
                }
            });
            guiPanel.add(slider);
        }

        {
            saveButton = new Button("Save");
            saveButton.setSize(40, 15);
            saveButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    saveModelClicked();
                }
            });
            saveButton.setEnabled(false);

            guiPanel.add(saveButton);
        }

        {
            applyButton = new Button("Apply");
            applyButton.setSize(40, 15);
            applyButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    applyClicked();
                }
            });
            applyButton.setEnabled(false);

            guiPanel.add(applyButton);
        }

        {
            Button cancelButton = new Button("Cancel");
            cancelButton.setSize(40, 15);
            cancelButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    cancelClicked();
                }
            });
            guiPanel.add(cancelButton);
        }

        refreshGUI();
        guiPanel.setVisible(true);
        guiPanel.requestFocus();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
        gdp = new GenericDialogPlus("CLIJx Weka Binary Pixel Classification");
        gdp.addMessage("Annotation");
        gdp.addButton("Add foreground", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                addForegroundClicked();
            }
        });
        gdp.addToSameRow();
        gdp.addButton("Add background", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                addBackgroundClicked();
            }
        });
        gdp.addToSameRow();
        gdp.addButton("Erase", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                eraseClicked();
            }
        });

        gdp.addMessage("Training");
        gdp.addStringField("Feature definition", CLIJxWekaPropertyHolder.pixelClassificationFeatureDefinition);

        gdp.addToSameRow();
        gdp.addButton("Train", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                trainClicked();
            }
        });

        gdp.addFileField("", CLIJxWekaPropertyHolder.pixelClassificationModelFile );

        gdp.addToSameRow();
        gdp.addButton("Save model", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                saveModelClicked();
            }
        });

        gdp.setModal(false);
        gdp.showDialog();
        gdp.setSize(new Dimension(500, 300));
*/
    }

    private void cancelClicked() {
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();
        inputImp.setProcessor(clijx.pull(clInput).getProcessor());
        cleanUp();
    }

    private void cleanUp()
    {
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();
        clijx.release(clInput);
        clInput = null;
        clijx.release(clResult);
        clResult = null;
        ImagePlus.removeImageListener(imageListener);
        inputImp.getWindow().removeMouseListener(mouseListener1);
        inputImp.getWindow().removeMouseMotionListener(mouseListener2);
        inputImp.getWindow().getCanvas().removeMouseMotionListener(mouseListener3);
        inputImp.setOverlay(null);
        inputImp.killRoi();
        guiPanel.setVisible(false);
        guiPanel.dispose();
        guiPanel = null;
        Roi.setColor(defaultRoiColor);
    }

    private void applyClicked() {
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();
        inputImp.setProcessor(clijx.pullBinary(clResult).getProcessor());



        Recorder.setCommand(null);
        boolean recordBefore = Recorder.record;
        Recorder.record = false;

        String inputImage = inputImp.getTitle();
        String outputImage = "CLIJxWeka_Pixel_Classified";

        recordIfNotRecorded("run", "\"CLIJ Macro Extensions\", \"cl_device=[" + clijx.getGPUName() + "]\"");
        recordIfNotRecorded("Ext.CLIJ_push", "\"" + inputImage + "\"");

        Recorder.recordString("Ext.CLIJx_applyOCLWekaModel(\"" +inputImage+ "\", \"" + outputImage + "\", \"" + CLIJxWekaPropertyHolder.pixelClassificationModelFile + "\");");

        recordIfNotRecorded("Ext.CLIJ_pull", "\"" + outputImage + "\"");

        Recorder.record = recordBefore;

        cleanUp();
    }

    private void recordIfNotRecorded(String recordMethod, String recordParameters) {
        if (Recorder.getInstance() == null) {
            return;
        }
        String text = Recorder.getInstance().getText();
        if (text.contains(recordMethod) && text.contains(recordParameters)) {
            return;
        }
        record(recordMethod, recordParameters);
    }

    private void record(String recordMethod, String recordParameters) {
        if (Recorder.getInstance() == null) {
            return;
        }
        Recorder.recordString(recordMethod + "(" + recordParameters + ");\n");
        Recorder.record = true;
    }


    private void refreshGUI() {
        final ImageWindow window = inputImp.getWindow();

        guiPanel.setSize(550, 30);
        guiPanel.setLocation(window.getX() + window.getCanvas().getX() - 1, window.getY() + window.getCanvas().getY() - 1);


        foregroundColorButton.setBackground(foregroundColor);
        backgroundColorButton.setBackground(backgroundColor);

        foregroundButton.setSelected(false);
        backgroundButton.setSelected(false);
        eraseButton.setSelected(false);

        foregroundButton.setBackground(null);
        backgroundButton.setBackground(null);
        eraseButton.setBackground(null);

        saveButton.setEnabled(clResult != null);
        applyButton.setEnabled(clResult != null);

        if (mouseMode == MouseMode.FOREGROUND) {
            Roi.setColor(foregroundColor);
            status.setStrokeColor(foregroundColor);
            foregroundButton.setBackground(foregroundColor);
            foregroundButton.setSelected(true);
        }
        if (mouseMode == MouseMode.BACKGROUND) {
            Roi.setColor(backgroundColor);
            backgroundButton.setBackground(backgroundColor);
            backgroundButton.setSelected(true);
        }
        if (mouseMode == MouseMode.ERASE) {
            Roi.setColor(eraseColor);
            eraseButton.setBackground(eraseColor);
            eraseButton.setSelected(true);
        }
        for (int i = 0; i < overlay.size(); i++) {
            Roi roi = overlay.get(i);
            if (roi.getName().compareTo(MouseMode.FOREGROUND.toString()) == 0) {
                roi.setStrokeColor(foregroundColor);
            } else if (roi.getName().compareTo(MouseMode.BACKGROUND.toString()) == 0) {
                roi.setStrokeColor(backgroundColor);
            }
        }
        guiPanel.validate();

    }

    private void updateVisualisation() {
        refreshGUI();
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();

        if (clResult != null) {
            ClearCLBuffer foreground = clijx.create(clInput.getDimensions(), NativeTypeEnum.UnsignedByte);
            ClearCLBuffer background = clijx.create(clInput.getDimensions(), NativeTypeEnum.UnsignedByte);
            clijx.multiplyImageAndScalar(clResult, foreground, 255);
            clijx.subtractImageFromScalar(foreground, background, 255);

            ClearCLBuffer rgb = clijx.create(new long[]{clInput.getWidth(), clInput.getHeight(), 3}, NativeTypeEnum.UnsignedByte);
            MakeRGB.makeRGB(clijx, clInput, foreground, background, rgb, (float) inputImp.getDisplayRangeMin(), (float) inputImp.getDisplayRangeMax(), foregroundColor, backgroundColor, overlayAlpha);

            ImageStack stack = clijx.pull(rgb).getStack();

            ImagePlus tempImp = new ImagePlus("temp", stack);
            tempImp = HyperStackConverter.toHyperStack(tempImp, 3, 1, 1);
            RGBStackConverter.convertToRGB(tempImp);

            inputImp.setProcessor(tempImp.getProcessor());

            clijx.release(foreground);
            clijx.release(background);
            clijx.release(rgb);

        } else {
            ImagePlus vis = clijx.pull(clInput);
            inputImp.setProcessor(vis.getProcessor());
        }
    }


    private void readPropertiesFromDialog() {
        //CLIJxWekaPropertyHolder.pixelClassificationFeatureDefinition = ((TextField)gdp.getStringFields().get(0)).getText();
        //CLIJxWekaPropertyHolder.pixelClassificationModelFile = ((TextField)gdp.getStringFields().get(1)).getText();
    }

    private void imageChanged() {
        System.out.println("image changed");
    }

    private void saveModelClicked() {
        if (clijxweka != null) {
            SaveDialog sd = new SaveDialog("Save model", CLIJxWekaPropertyHolder.pixelClassificationModelFile, ".model");
            String filename = sd.getFileName();
            if (filename != null) {
                clijxweka.saveClassifier(filename);
            }
        }
    }

    private void trainClicked() {
        readPropertiesFromDialog();

        ImagePlus groundTruth = NewImage.createFloatImage("ground_truth", inputImp.getWidth(), inputImp.getHeight(), 1, NewImage.FILL_BLACK);
        for (int i = 0; i < overlay.size(); i++) {
            Roi roi = overlay.get(i);
            if (roi instanceof PolygonRoi) {
                String name = roi.getName();
                roi = Selection.lineToArea(roi);
                groundTruth.setRoi(roi);
                IJ.run(groundTruth, "Multiply...", "value=0");
                IJ.run(groundTruth, "Add...", "value=" + (name.compareTo(MouseMode.FOREGROUND.toString()) == 0 ? 2 : 1));
            }
        }
//        groundTruth.show();
  //      groundTruth.setDisplayRange(0, 2);

        //if (true) {return;}
        CLIJx clijx = CLIJxWekaPropertyHolder.getCLIJx();



        ClearCLBuffer clGroundTruth = clijx.push(groundTruth);

        if (clResult != null) {
            clijx.release(clResult);
        }
        clResult = clijx.create(clGroundTruth);

        ClearCLBuffer clFeatureStack = GenerateFeatureStack.generateFeatureStack(clijx, clInput, CLIJxWekaPropertyHolder.pixelClassificationFeatureDefinition);

        clijxweka = TrainWekaModel.trainWekaModel(clijx, clFeatureStack, clGroundTruth, CLIJxWekaPropertyHolder.pixelClassificationModelFile);

        String ocl = clijxweka.getOCL();

        ApplyOCLWekaModel.applyOCL(clijx, clFeatureStack, clResult, ocl);

        updateVisualisation();

        clijx.release(clGroundTruth);
        clijx.release(clFeatureStack);
        //clijx.release(clResult);
    }



    enum MouseMode {
        FOREGROUND,
        BACKGROUND,
        ERASE
    };
    MouseMode mouseMode = MouseMode.FOREGROUND;

    private void eraseClicked() {
        IJ.setTool("freeline");
        mouseMode = MouseMode.ERASE;
        Roi.setColor(eraseColor);
    }
    private void addBackgroundClicked() {
        IJ.setTool("freeline");
        mouseMode = MouseMode.BACKGROUND;
        Roi.setColor(backgroundColor);
    }
    private void addForegroundClicked() {
        IJ.setTool("freeline");
        mouseMode = MouseMode.FOREGROUND;
        Roi.setColor(foregroundColor);
    }
    private void mouseUp(MouseEvent e) {
        System.out.println("mouse up" + e.isControlDown());
        Roi roi = inputImp.getRoi();
        if (roi != null) {
            if (mouseMode == MouseMode.FOREGROUND || mouseMode == MouseMode.BACKGROUND) {
                //roi = Selection.lineToArea(roi);
                roi.setStrokeWidth(2);
                if ((mouseMode == MouseMode.FOREGROUND && !e.isControlDown()) || (mouseMode != MouseMode.FOREGROUND && e.isControlDown())){
                    roi.setName(MouseMode.FOREGROUND.toString());
                    roi.setStrokeColor(foregroundColor);
                } else {
                    roi.setName(MouseMode.BACKGROUND.toString());
                    roi.setStrokeColor(backgroundColor);
                }
                overlay.add(roi);
            } else {
                System.out.println("Overlay size "+  overlay.size());
                for (int i = overlay.size() - 1; i >= 0; i--) {
                    Roi otherRoi = overlay.get(i);
                    if (otherRoi instanceof PolygonRoi) {
                        PolygonRoi polyline1 = (PolygonRoi) roi;
                        PolygonRoi polyline2 = (PolygonRoi) otherRoi;
                        if (linesIntersect(polyline1, polyline2)) {
                            overlay.remove(polyline2);
                        }
                    }
                }
            }
        }
        inputImp.killRoi();
        inputImp.setOverlay(overlay);
    }

    private boolean linesIntersect(PolygonRoi polyline1, PolygonRoi polyline2) {
        FloatPolygon polygon1 = polyline1.getFloatPolygon();
        FloatPolygon polygon2 = polyline2.getFloatPolygon();

        for (int i = 1; i < polygon1.npoints; i++) {
            for (int j = 1; j < polygon2.npoints; j++) {
                double ax1 = polygon1.xpoints[i];
                double ay1 = polygon1.ypoints[i];
                double ax2 = polygon1.xpoints[i - 1];
                double ay2 = polygon1.ypoints[i - 1];

                double bx1 = polygon2.xpoints[j];
                double by1 = polygon2.ypoints[j];
                double bx2 = polygon2.xpoints[j - 1];
                double by2 = polygon2.ypoints[j - 1];

                if (Line2D.linesIntersect(ax1, ax1, ax2, ay2, bx1, by1, bx2, by2)) {
                    return true;
                }
            }
        }
        return false;
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
        /*ClearCLBuffer slice1 = clijx.create(input);
        ClearCLBuffer slice2 = clijx.create(input);
        ClearCLBuffer slice3 = clijx.create(input);

        clijx.flip(input, slice1, true, false);
        clijx.flip(input, slice2, true, true);
        clijx.flip(input, slice3, false, true);

        ClearCLBuffer */

        clijx.show(input,"input");

        new CLIJxWekaBinaryPixelClassification().run(null);
    }


}
