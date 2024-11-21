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

import java.awt.Color;
import java.awt.Component;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import javax.swing.Icon;
import javax.swing.UIManager;

import com.formdev.flatlaf.FlatLaf.DisabledIconProvider;;

/**
 * Unicode symbol as Icon
 */
public class SymbolIcon implements Icon, DisabledIconProvider {
	private String symbol = null;
	private int size;

	/**
	 * 
	 */
	public SymbolIcon(String symbol) {
		this(symbol, 16);
	}

	public SymbolIcon(String symbol, int size) {
		this.symbol = symbol;
		this.size = size;
	}

	@Override
	public int getIconHeight() {
		return size;
	}

	@Override
	public int getIconWidth() {
		return size;
	}

	@Override
	public void paintIcon(Component c, Graphics g, int x, int y) {
		final Font oldFont = g.getFont();
		final float fs = size/1.1f;
		g.setFont(oldFont.deriveFont(fs));
		final Color color = UIManager.getColor(c.isEnabled() ? "textText" : "textInactiveText");
		g.setColor(color);
		if (g instanceof Graphics2D)
			((Graphics2D) g).setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
					RenderingHints.VALUE_TEXT_ANTIALIAS_LCD_HRGB);
		g.drawString(symbol, x, y+(int)fs);
		g.setFont(oldFont);
	}

	@Override
	public Icon getDisabledIcon() {
		return this;
	}
}
