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
package trainableSegmentation.filters;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.OvalRoi;
import ij.plugin.filter.PlugInFilter;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

/**
 * This class implements a circular entropy filter
 * 
 */
public class Entropy_Filter implements PlugInFilter
{
	/** original image */
	private ImagePlus origImg = null;
	/** radius to use (in pixels) */
	private int radius = 2;
	/** number of bins to use in the histogram */
	private int numBins = 256;
	
	/**
	 * Main method when called as a plugin filter
	 * @param ip input image
	 */
	public void run(ImageProcessor ip) 
	{
		applyEntropy(ip, radius, numBins);			
	}

	/**
	 * Setup method
	 * @param arg filter arguments
	 * @param imp input image
	 */
	public int setup(String arg, ImagePlus imp)
	{
		if (arg.equals("about"))
		{
			showAbout();
			return DONE;
		}

		if (imp == null)
			return DONE;

		GenericDialog gd = new GenericDialog("Entropy filter");
		gd.addNumericField("Radius (in pixels)", 3, 0);
		gd.addNumericField("Number of bins", 256, 0);

		gd.showDialog();
		if ( gd.wasCanceled() )
			return DONE;

		radius = (int) gd.getNextNumber();
		numBins = (int) gd.getNextNumber();
		if( numBins <= 0 || numBins >= 256)
			numBins = 256;

		return DOES_ALL+DOES_STACKS+PARALLELIZE_STACKS;
	}




	/**
	 * Get the entropy filter version of an image
	 * @param ip input image
	 * @param radius radius to use (in pixels)
	 * @param numBins number of bins to use in the histogram
	 * @return entropy image (32-bit)
	 */
	public FloatProcessor getEntropy(
			ImageProcessor ip, 
			int radius,
			int numBins)
	{
		final double log2=Math.log(2.0);
		ip.resetMinAndMax();
		final ByteProcessor bp = (ByteProcessor) ip.convertToByte(true);
		
		bp.setHistogramRange( 0, 255 );
		bp.setHistogramSize( numBins );
		
		final FloatProcessor fp = new FloatProcessor(bp.getWidth(), bp.getHeight());
		
		final int size = 2 * radius + 1;
		
		for(int i=0; i<bp.getWidth(); i++)
		{			
			for(int j=0; j<bp.getHeight(); j++)
			{
				final OvalRoi roi = new OvalRoi(i-radius, j-radius, size, size);				
				bp.setRoi( roi );
				final int[] histogram = bp.getHistogram(); // Get histogram from the ROI
				
				double total = 0;
				for (int k = 0 ; k < numBins ; k++ )
					total +=histogram[ k ];

				double entropy = 0;
				for (int k = 0 ; k < numBins ; k++ )
				{
					if (histogram[k]>0)
					{   
						double p = histogram[k]/total; // calculate p
		  				entropy += -p * Math.log(p)/log2;						
					}
				}
				fp.putPixelValue(i, j, entropy );
			}
		}
		
		return fp;
	}
	
	/**
	 * Apply entropy filter to an image
	 * @param ip input image
	 * @param radius radius to use (in pixels)
	 * @param numBins number of bins to use in the histogram
	 */
	public void applyEntropy(
			ImageProcessor ip, 
			int radius, 
			int numBins)
	{
		final FloatProcessor fp = getEntropy(ip, radius, numBins);
		
		ImageProcessor ip2;
		
		// rescale to the corresponding number of bits
		if (!(ip instanceof FloatProcessor))
		{
			if (ip instanceof ByteProcessor)	
			{
				ip2 = fp.convertToByte(true);
			}
			else
				ip2 = fp.convertToShort(true);
			ip.setPixels(ip2.getPixels());
		}
		else
			ip.setPixels( fp.getPixels() );
		
		ip.resetMinAndMax();
		
	}

	/**
	 * Display filter information
	 */
	void showAbout() 
	{
		IJ.showMessage("Entropy filter...",
				"Circular entropy filter by I. Arganda-Carreras\n"+
				"ImageJ local entropy filter. Output is 32-bit\n");
	}
	
}
