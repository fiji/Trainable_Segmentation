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

/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
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
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), Verena Kaynig (verena.kaynig@inf.ethz.ch),
 *          Albert Cardona (acardona@ini.phys.ethz.ch)
 */

import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.concurrent.*;

/**
 * This class stores the feature stacks of a set of input slices.
 * It can be used so for 2D stacks or as the container of 3D features (by
 * using a feature stack per section). 
 * 
 * @author Ignacio Arganda-Carreras (iarganda@mit.edu)
 *
 */
public class FeatureStackArray 
{
	/** array of feature stacks */
	private FeatureStack[] featureStackArray;
	
	/** index of the feature stack that is used as reference (to read attribute, etc.).
	 * -1 if not defined yet. */
	private int referenceStackIndex = -1;
	
	/** minimum sigma/radius used in the filters */
	private float minimumSigma = 1;
	/** maximum sigma/radius used in the filters */
	private float maximumSigma = 16;
	/** use neighborhood flag */
	private boolean useNeighbors = false;
	/** expected membrane thickness (in pixels) */
	private int membraneThickness = 1;	
	/** size of the patch to use to enhance membranes (in pixels, NxN) */
	private int membranePatchSize = 19;
	/** common enabled features */
	private boolean[] enabledFeatures = null;
	
	/** flag to specify the use of the old color format (using directly the RGB values as float) */
	private boolean oldColorFormat = false;
	/** flag to specify the use of the old (wrong) Hessian format (fixed in
	 * version 3.2.25 of TWS) */
	private boolean oldHessianFormat = false;
	/**
	 * Initialize a feature stack list of a specific size (with default values
	 * for the rest of parameters).
	 *
	 * @param num number of elements in the list
	 */
	public FeatureStackArray( final int num )
	{
		this.featureStackArray = new FeatureStack[ num ];
	}

	/**
	 * Initialize a feature stack list of a specific size
	 * 
	 * @param num number of elements in the list
	 * @param minimumSigma minimum sigma value (usually filter radius)
	 * @param maximumSigma maximum sigma value (usually filter radius)
	 * @param useNeighbors flag to use neighbor features
	 * @param membraneSize expected membrane thickness
	 * @param membranePatchSize membrane patch size
	 * @param enabledFeatures array of flags to enable features
	 */
	public FeatureStackArray(
			final int num,
			final float minimumSigma,
			final float maximumSigma,
			final boolean useNeighbors,
			final int membraneSize,
			final int membranePatchSize,
			final boolean[] enabledFeatures)
	{
		this.featureStackArray = new FeatureStack[num]; 
		this.minimumSigma = minimumSigma;
		this.maximumSigma = maximumSigma;
		this.useNeighbors = useNeighbors;
		this.membraneThickness = membraneSize;
		this.membranePatchSize = membranePatchSize;
		this.enabledFeatures = enabledFeatures;
	}
	
	/**
	 * Create a feature stack array based on specific filters
	 * 
	 * @param inputImage original image
	 * @param filters stack of filters to apply to the original image in order to create the features
	 */
	public FeatureStackArray(
			final ImagePlus inputImage,
			final ImagePlus filters)
	{
		this.featureStackArray = new FeatureStack[ inputImage.getImageStackSize() ];
		
		
		for(int i=1; i <= featureStackArray.length; i++)
		{		
			featureStackArray[ i-1 ] = new FeatureStack(new ImagePlus("slice " + i, inputImage.getImageStack().getProcessor(i)));			
			featureStackArray[ i-1 ].addFeaturesMT( filters );			 
		}
	}
	
	/**
	 * Get the number of feature stacks
	 * 
	 * @return number of feature stacks stored in the array
	 */
	public int getSize()
	{
		return this.featureStackArray.length;
	}
	
	/**
	 * Get n-th stack in the array (remember n&gt;=0)
	 * @param n position of the stack to get
	 * @return feature stack of the corresponding slice
	 */
	public FeatureStack get(int n)
	{
		return featureStackArray[n];
	}
	/**
	 * Return if image stacks are RGB or grayscale.
	 * @return true if image feature stacks are RGB, false if grayscale
	 */
	public boolean isRGB()
	{
		if( this.referenceStackIndex != -1 )
			return this.featureStackArray[ referenceStackIndex ]
					.getStack().getBitDepth() == 24;
		else
		{
			IJ.log("Warning. Error in FeatureStackArray: trying to access empty array!");
			return false;
		}
	}
	/**
	 * Set a member of the list
	 * @param fs new feature stack  
	 * @param index index of the new feature stack in the array
	 */
	public void set(FeatureStack fs, int index)
	{
		this.featureStackArray[ index ] = fs;
		if( this.referenceStackIndex == -1 )
			setReference( index );
	}
	
	/**
	 * Update specific feature stacks in the list (multi-thread fashion)
	 * 
	 * @param update boolean array indicating which feature stack to update
	 * @return false if any feature stack was not properly updated
	 */
	public boolean updateFeaturesMT(boolean[] update)
	{
		if (Thread.currentThread().isInterrupted() )
			return false;
		
		final int numProcessors = Prefs.getThreads();
		final ExecutorService exe = Executors.newFixedThreadPool( numProcessors );
		
		final ArrayList< Future<Boolean> > futures = new ArrayList< Future<Boolean> >();
		
		try{
			for(int i=0; i<featureStackArray.length; i++)
			{
				if(null != featureStackArray[i])
					if(update[i])
					{
						IJ.log("Updating features of slice number " + (i+1) + "...");						
						featureStackArray[i].setEnabledFeatures(enabledFeatures);
						featureStackArray[i].setMembranePatchSize(membranePatchSize);
						featureStackArray[i].setMembraneSize(membraneThickness);
						featureStackArray[i].setMaximumSigma(maximumSigma);
						featureStackArray[i].setMinimumSigma(minimumSigma);
						featureStackArray[i].setUseNeighbors(useNeighbors);
						featureStackArray[i].setOldColorFormat(oldColorFormat);
						featureStackArray[i].setOldHessianFormat(oldHessianFormat);
						if ( featureStackArray.length == 1 )
						{
							if( !featureStackArray[i].updateFeaturesMT() )
								return false;							
						}
						else
							futures.add(exe.submit( updateFeatures( featureStackArray[i] ) ));

						if(referenceStackIndex == -1)
							this.referenceStackIndex = i;
					}
			}
			
			// Wait for the jobs to be done
			for(Future<Boolean> f : futures)
			{
				final boolean result = f.get();
				if(!result)
					return false;
			}			
		}
		catch (InterruptedException e) 
		{
			IJ.log("The feature update was interrupted by the user.");
			exe.shutdownNow();
			return false;
		}
		catch(Exception ex)
		{
			IJ.log("Error when updating feature stack array.");
			ex.printStackTrace();
			exe.shutdownNow();
			return false;
		}
		finally{
			exe.shutdown();
		}	
		
		
		return true;
	}

	/**
	 * Update all feature stacks in the list (multi-thread fashion) 
	 * @return false if error, true otherwise
	 */
	public boolean updateFeaturesMT() {
		final int numProcessors = Prefs.getThreads();
		final ExecutorService exe = Executors.newFixedThreadPool(numProcessors);
		final ArrayList<Future<Boolean>> futures = new ArrayList<Future<Boolean>>();

		IJ.showStatus("Updating features...");

		try {
			for (int i = 0; i < featureStackArray.length; i++) {
				if (null != featureStackArray[i]) {
					IJ.log("Updating features of slice number " + (i + 1) + "...");
					setupFeatureStack(featureStackArray[i]);

					if (featureStackArray.length == 1) {
						if (!updateFeatureStack(featureStackArray[i]))
							return false;
					} else {
						int finalI = i;
						futures.add(exe.submit(() -> updateFeatureStack(featureStackArray[finalI])));
					}

					if (referenceStackIndex == -1)
						this.referenceStackIndex = i;
				}
			}

			// Wait for the jobs to be done
			waitForFeaturesToComplete(futures);

		} catch (InterruptedException e) {
			handleInterruptedException(e);
			exe.shutdownNow();
			return false;
		} catch (Exception ex) {
			handleFeatureStackUpdateError(ex);
			exe.shutdownNow();
			return false;
		} finally {
			exe.shutdown();
		}

		return true;
	}

	private void setupFeatureStack(FeatureStack featureStack) {
		featureStack.setEnabledFeatures(enabledFeatures);
		featureStack.setMembranePatchSize(membranePatchSize);
		featureStack.setMembraneSize(membraneThickness);
		featureStack.setMaximumSigma(maximumSigma);
		featureStack.setMinimumSigma(minimumSigma);
		featureStack.setUseNeighbors(useNeighbors);
		featureStack.setOldColorFormat(oldColorFormat);
		featureStack.setOldHessianFormat(oldHessianFormat);
	}

	private boolean updateFeatureStack(FeatureStack featureStack) {
		return featureStack.updateFeaturesMT();
	}

	private void waitForFeaturesToComplete(ArrayList<Future<Boolean>> futures) throws InterruptedException, ExecutionException {
		int currentIndex = 0;
		final int finalIndex = featureStackArray.length;

		for (Future<Boolean> f : futures) {
			final boolean result = f.get();
			currentIndex++;
			IJ.showStatus("Updating features...");
			IJ.showProgress(currentIndex, finalIndex);
			if (!result)
				throw new ExecutionException("One of the feature stack update jobs failed", null);
		}
	}

	private void handleInterruptedException(InterruptedException e) {
		IJ.log("The feature update was interrupted by the user.");
		IJ.showStatus("The feature update was interrupted by the user.");
		IJ.showProgress(1.0);
	}

	private void handleFeatureStackUpdateError(Exception ex) {
		IJ.log("Error when updating feature stack array.");
		ex.printStackTrace();
	}



	/**
	 * Update features of a feature stack (to be submitted to an Executor Service)
	 * 
	 * @param fs feature stack to be updated
	 * @return true if everything went correct
	 */
	public Callable<Boolean> updateFeatures(
			final FeatureStack fs)
	{
		if (Thread.currentThread().isInterrupted()) 
			return null;
		
		return new Callable<Boolean>(){
			public Boolean call(){
				return fs.updateFeaturesST();
			}
		};
	}
	
	
	/**
	 * Reset the reference index (used when the are 
	 * changes in the features)
	 */
	public void resetReference()
	{
		this.referenceStackIndex = -1;
	}
	
	/**
	 * Set the reference index (used when the are 
	 * changes in the features).
	 * @param index reference index
	 */
	public void setReference( int index )
	{
		this.referenceStackIndex = index;
	}
	
	/**
	 * Shut down the executor service
	 */
	public void shutDownNow()
	{
		for(int i=0; i<featureStackArray.length; i++)
			if(null != featureStackArray[i])
			{
				featureStackArray[i].shutDownNow();
			}
	}
	
	/**
	 * Check if the array has not been yet initialized
	 * 
	 * @return true if the array has been initialized
	 */
	public boolean isEmpty() 
	{
		for(int i=0; i<getSize(); i++)
			if( null != featureStackArray[i] && featureStackArray[i].getSize()>1 )
				return false;
		return true;
	}

	/**
	 * Get the number of features of the reference stack (consistent all along the array)
	 * @return number of features on each feature stack of the array
	 */
	public int getNumOfFeatures() {
		if(referenceStackIndex == -1)
			return -1;
		return featureStackArray[referenceStackIndex].getSize();
	}
	
	/**
	 * Get a specific label of the reference stack
	 * @param index slice index (&gt;=1)
	 * @return label name
	 */
	public String getLabel(int index)
	{
		if(referenceStackIndex == -1)
			return null;
		return featureStackArray[referenceStackIndex].getSliceLabel(index);
	}
	
	/**
	 * Get the features enabled for the reference stack
	 * @return features to be calculated on each stack
	 */
	public boolean[] getEnabledFeatures()
	{
		if(referenceStackIndex == -1)
			return this.enabledFeatures;
		return featureStackArray[referenceStackIndex].getEnabledFeatures();
	}

	/**
	 * Set the features enabled for the reference stack
	 * @param newFeatures boolean flags for the features to use
	 */
	public void setEnabledFeatures(boolean[] newFeatures) 
	{
		this.enabledFeatures = newFeatures;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setEnabledFeatures(newFeatures);
	}

	/**
	 * Set expected membrane thickness
	 * 
	 * @param thickness expected membrane thickness in pixels
	 */
	public void setMembraneSize(int thickness) 
	{
		this.membraneThickness = thickness;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setMembraneSize(thickness);
	}

	/**
	 * Set membrane patch size.
	 * @param patchSize membrange patch size in pixels
	 */
	public void setMembranePatchSize(int patchSize) 
	{
		this.membranePatchSize = patchSize;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setMembranePatchSize(patchSize);
	}

	/**
	 * Set maximum sigma.
	 * @param sigma maximum sigma
	 */
	public void setMaximumSigma(float sigma) 
	{
		this.maximumSigma = sigma;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setMaximumSigma(sigma);		
	}

	/**
	 * Set minimum sigma.
	 * @param sigma minimum sigma
	 */
	public void setMinimumSigma(float sigma) 
	{
		this.minimumSigma = sigma;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setMinimumSigma(sigma);
	}

	/**
	 * Set the use of neighbor features.
	 * @param useNeighbors
	 */
	public void setUseNeighbors(boolean useNeighbors) 
	{
		this.useNeighbors = useNeighbors;
		if(referenceStackIndex != -1)
			featureStackArray[referenceStackIndex].setUseNeighbors(useNeighbors);
	}

	/**
	 * Check if neighbor features are used.
	 * @return true if neighbor features are used
	 */
	public boolean useNeighborhood() {
		if(referenceStackIndex != -1)
			return featureStackArray[referenceStackIndex].useNeighborhood();
		return useNeighbors;
	}
	
	/**
	 * Get the index of the reference slice in the feature stack array.
	 * @return index of the reference slice in the feature stack array
	 */
	public int getReferenceSliceIndex()
	{
		return referenceStackIndex;
	}
	
	/**
	 * Get the width of the feature stacks.
	 * @return width of the feature stacks or -1 if not initialized yet.
	 */
	public int getWidth()
	{
		if(referenceStackIndex != -1)
			return featureStackArray[referenceStackIndex].getWidth();
		return -1;
	}
	
	/**
	 * Get the height of the feature stacks.
	 * @return height of the feature stacks or -1 if not initialized yet.
	 */
	public int getHeight()
	{
		if(referenceStackIndex != -1)
			return featureStackArray[referenceStackIndex].getHeight();
		return -1;
	}
	
	/**
	 * Set the use of the old color format.
	 * @param oldColorFormat true if old color format is to be used
	 */
	public void setOldColorFormat( boolean oldColorFormat ) 
	{
		this.oldColorFormat = oldColorFormat;
		if( referenceStackIndex != -1 )
			featureStackArray[ referenceStackIndex ].setOldColorFormat( oldColorFormat );
	}
	
	/**
	 * Check if the old color format is used.
	 * @return true if the old color format is used
	 */
	public boolean isOldColorFormat()
	{
		return this.oldColorFormat;
	}
	/**
	 * Specify the use of the old (wrong) Hessian format (fixed in
	 * version 3.2.25 of TWS)
	 * @param oldHessianFormat true if old Hessian format is to be used
	 */
	public void setOldHessianFormat( boolean oldHessianFormat )
	{
		this.oldHessianFormat = oldHessianFormat;
		if( referenceStackIndex != -1 )
			featureStackArray[ referenceStackIndex ]
					.setOldHessianFormat( oldHessianFormat );
	}
	
	/**
	 * Check if the old Hessian format is used.
	 * @return true if the old Hessian format is used
	 */
	public boolean isOldoldHessianFormatFormat()
	{
		return this.oldHessianFormat;
	}
	/**
	 * Reorder the features of each stack based on the order of attributes given
	 * by a set of instances.
	 * @param data set of instances to get the order from
	 * @return true if reordering was possible
	 */
	public boolean reorderFeatures( Instances data )
	{
		if( null == data )
			return false;
		for(int i=0; i<featureStackArray.length; i++)
			if(null != featureStackArray[i])
			{
				if( ! featureStackArray[i].reorderFeatures(data) );
					return false;
			}
		return true;
	}
}

	
