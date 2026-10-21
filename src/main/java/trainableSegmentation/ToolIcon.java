/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2024 Fiji developers.
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
package trainableSegmentation;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import javax.swing.Icon;
import javax.swing.UIManager;

/**
 * 16x16 vector icons drawn with Java2D (no font or image resources needed)
 */
public class ToolIcon implements Icon {
	public enum Kind { SAVE, LOAD, TRASH, PEN, PALETTE, ARROW }

	private static final int SIZE = 16;
	private final Kind kind;
	private final boolean dimmed;

	public ToolIcon(Kind kind) {
		this(kind, false);
	}

	/** @param dimmed always paint translucent, for use as an explicit disabled icon */
	public ToolIcon(Kind kind, boolean dimmed) {
		this.kind = kind;
		this.dimmed = dimmed;
	}

	@Override
	public int getIconWidth() {
		return SIZE;
	}

	@Override
	public int getIconHeight() {
		return SIZE;
	}

	@Override
	public void paintIcon(Component c, Graphics g, int x, int y) {
		final Graphics2D g2 = (Graphics2D) g.create();
		g2.translate(x, y);
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
		// disabled: translucent foreground, visible on any look and feel (menu items included)
		// don't use c.getForeground(): menu items report a near-invisible colour when disabled
		final Color fg = UIManager.getColor("Label.foreground") == null ? Color.DARK_GRAY
				: UIManager.getColor("Label.foreground");
		g2.setColor(c.isEnabled() && !dimmed ? fg : new Color(fg.getRed(), fg.getGreen(), fg.getBlue(), 110));
		g2.setStroke(new BasicStroke(1.2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
		switch (kind) {
		case SAVE: // floppy disk
			g2.drawRoundRect(2, 2, 12, 12, 2, 2);
			g2.fillRect(5, 2, 6, 4); // shutter
			g2.drawRect(4, 9, 8, 5); // label
			break;
		case LOAD: // open folder
			g2.drawPolyline(new int[] { 2, 2, 6, 8, 13 }, new int[] { 13, 3, 3, 5, 5 }, 5);
			g2.fillPolygon(new int[] { 1, 4, 15, 12 }, new int[] { 14, 7, 7, 14 }, 4);
			break;
		case TRASH:
			g2.drawLine(2, 4, 14, 4); // lid
			g2.drawRect(6, 2, 4, 2); // handle
			g2.drawPolygon(new int[] { 3, 13, 12, 4 }, new int[] { 5, 5, 14, 14 }, 4);
			g2.drawLine(6, 7, 6, 12);
			g2.drawLine(8, 7, 8, 12);
			g2.drawLine(10, 7, 10, 12);
			break;
		case PEN:
			g2.setStroke(new BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER));
			g2.drawLine(12, 3, 6, 9); // body
			g2.setStroke(new BasicStroke(1f));
			g2.fillPolygon(new int[] { 5, 8, 2 }, new int[] { 8, 11, 14 }, 3); // tip
			break;
		case ARROW: // small down arrow of a dropdown button
			g2.fillPolygon(new int[] { 4, 12, 8 }, new int[] { 6, 6, 11 }, 3);
			break;
		case PALETTE:
			g2.setColor(Color.RED);
			g2.fillOval(1, 5, 8, 8);
			g2.setColor(Color.GREEN);
			g2.fillOval(7, 5, 8, 8);
			g2.setColor(Color.BLUE);
			g2.fillOval(4, 1, 8, 8);
			break;
		}
		g2.dispose();
	}
}
