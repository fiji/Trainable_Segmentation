package net.haesleinhuepf.clijx.weka.gui;

import ij.ImageListener;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Roi;

import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class InteractivePanelPlugin {


    Color defaultRoiColor;

    protected Dialog guiPanel;
    ImageWindow window;

    private ImageListener imageListener;
    private MouseAdapter mouseListener1;
    private MouseAdapter mouseListener2;
    private MouseAdapter mouseListener3;

    protected void attach(ImageWindow window) {
        defaultRoiColor = Roi.getColor();
        this.window = window;
        guiPanel = new Dialog(window);
        //guiPanel.setLayout(new GridLayout(1, 7));
        guiPanel.setLayout(new GridBagLayout());
        guiPanel.setUndecorated(true);

        imageListener = new ImageListener() {
            @Override
            public void imageOpened(ImagePlus imp) {

            }

            @Override
            public void imageClosed(ImagePlus imp) {

            }

            @Override
            public void imageUpdated(ImagePlus imp) {
                if (imp.getWindow() == window) {
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
        window.getCanvas().addMouseListener(mouseListener1);

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
                        refresh();
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

        guiPanel.setVisible(true);
        guiPanel.requestFocus();
    }

    protected void refresh() {
//        guiPanel.setSize(550, 30);
  //      guiPanel.setLocation(window.getX() + window.getCanvas().getX() - 1, window.getY() + window.getCanvas().getY() - 1);

        guiPanel.setSize(window.getCanvas().getWidth() + 2, 30);
        guiPanel.setLocation(window.getX() + window.getCanvas().getX() - 1, window.getY() + window.getCanvas().getY() - 1);

        guiPanel.validate();
    }

    protected void dismantle() {

        ImagePlus.removeImageListener(imageListener);
        window.removeMouseListener(mouseListener1);
        window.removeMouseMotionListener(mouseListener2);
        window.getCanvas().removeMouseMotionListener(mouseListener3);

        Roi.setColor(defaultRoiColor);
    }

    protected void mouseUp(MouseEvent e) {
    }

    protected void imageChanged() {
    }

}
