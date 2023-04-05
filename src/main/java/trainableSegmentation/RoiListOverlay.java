/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2023 Fiji developers.
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */
/**
 * Trainable_Segmentation plug-in for ImageJ and Fiji.
 * 2010 Ignacio Arganda-Carreras
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation (http://www.gnu.org/licenses/gpl.txt )
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 */

package trainableSegmentation;

import fiji.util.gui.OverlayedImageCanvas.Overlay;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.gui.ShapeRoiHelper;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Composite;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.Stroke;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RoiListOverlay implements Overlay {
	private ArrayList<Roi> roi = null;
	private Color color = Roi.getColor();
	private Composite composite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER);
	private Map<Integer, RoiPainter> painterMap = new HashMap<Integer, RoiPainter>();

	public RoiListOverlay() {
		initializePainterMap();
	}

	public RoiListOverlay(ArrayList<Roi> roi, Composite composite, Color color) {
		setRoi(roi);
		setComposite(composite);
		setColor(color);
		initializePainterMap();
	}

	private void initializePainterMap() {
		painterMap.put(Roi.FREELINE, new LinePainter());
		painterMap.put(Roi.LINE, new LinePainter());
		painterMap.put(Roi.POLYLINE, new LinePainter());
		painterMap.put(Roi.RECTANGLE, new FillPainter());
		painterMap.put(Roi.OVAL, new FillPainter());
		painterMap.put(Roi.POLYGON, new FillPainter());
		painterMap.put(Roi.FREEROI, new FillPainter());
		painterMap.put(Roi.TRACED_ROI, new FillPainter());
		painterMap.put(Roi.FREELINE, new LinePainter());
		painterMap.put(Roi.LINE, new LinePainter());
		painterMap.put(Roi.POLYLINE, new LinePainter());
		painterMap.put(Roi.POINT, new FillPainter());
	}

	public void paint(Graphics g, int x, int y, double magnification) {
		if (null == this.roi)
			return;
		// Set ROI image to null to avoid repainting
		for (Roi r : this.roi) {
			r.setImage(null);
			Shape shape = ShapeRoiHelper.getShape(new ShapeRoi(r));
			final Rectangle roiBox = r.getBounds();
			final Graphics2D g2d = (Graphics2D) g;
			final Stroke originalStroke = g2d.getStroke();
			final AffineTransform originalTransform = g2d.getTransform();
			final AffineTransform at = new AffineTransform();
			at.scale(magnification, magnification);
			at.translate(roiBox.x - x, roiBox.y - y);
			at.concatenate(originalTransform);

			g2d.setTransform(at);
			final Composite originalComposite = g2d.getComposite();
			g2d.setComposite(this.composite);
			g2d.setColor(this.color);

			RoiPainter painter = painterMap.get(r.getType());
			painter.draw(g2d, r);

			g2d.setTransform(originalTransform);
			g2d.setComposite(originalComposite);
			g2d.setStroke(originalStroke);
		}

	}

	public void setRoi(ArrayList<Roi> roi) {
		this.roi = roi;
	}

	public void setComposite(Composite composite) {
		this.composite = composite;
	}

	public void setColor(Color color) {
		this.color = color;
	}

	public String toString() {
		return "RoiOverlay(" + roi + ")";
	}
}