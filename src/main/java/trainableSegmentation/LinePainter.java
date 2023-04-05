package trainableSegmentation;

import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.gui.ShapeRoiHelper;

import java.awt.*;

public class LinePainter implements RoiPainter {
    @Override
    public void draw(Graphics2D g2d, Roi roi) {
        Shape shape = ShapeRoiHelper.getShape(new ShapeRoi(roi));
        g2d.draw(shape);
    }
}
