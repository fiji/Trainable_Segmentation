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
package ai;

import java.io.Serializable;
import java.util.ArrayList;

import weka.core.Instance;
import weka.core.Instances;

public abstract class SplitFunction implements Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected int index;
	protected double threshold;
	protected boolean allSame;
	public abstract void init(final Instances data, final ArrayList<Integer> indices);
	public abstract boolean evaluate(final Instance instance);
	public abstract SplitFunction newInstance();
}
