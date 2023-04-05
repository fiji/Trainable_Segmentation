package trainableSegmentation;

import ij.gui.Roi;

import java.awt.*;

public interface RoiPainter {
    void draw(Graphics2D g2d, Roi roi);
}
