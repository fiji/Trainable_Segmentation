package trainableSegmentation;

import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.gui.ShapeRoiHelper;

import java.awt.*;

public class FillPainter implements RoiPainter {
    @Override
    public void draw(Graphics2D g2d, Roi roi) {
        Shape shape = ShapeRoiHelper.getShape(new ShapeRoi(roi));
        g2d.fill(shape);
    }
}
