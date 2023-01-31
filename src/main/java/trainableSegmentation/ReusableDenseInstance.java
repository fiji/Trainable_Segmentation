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
package trainableSegmentation;

import weka.core.DenseInstance;

/**
 * This class extends the WEKA DenseInstance so it can be reused. This is an 
 * important feature to save time and memory when classifying iteratively many 
 * new instances.
 * 
 * @author Ignacio Arganda-Carreras (iargandacarreras@gmail.com)
 *
 */
public class ReusableDenseInstance extends DenseInstance{

	/** for serialization */
	private static final long serialVersionUID = -2322621365960366756L;

	/**
	 * Construct reusable dense instance
	 * 
	 * @param weight instance weight
	 * @param attValues array of attribute values
	 */
	public ReusableDenseInstance(double weight, double[] attValues) {
		super(weight, attValues);
	}

	/**
	 * Set instance values
	 * 
	 * @param weight instance weight
	 * @param attValues array of attribute values
	 */
	public void setValues( double weight, double[] attValues )
	{
		super.m_Weight = weight;
		super.m_AttValues = attValues;
	}
	
	
}
