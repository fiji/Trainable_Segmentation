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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.Test;

import ij.ImagePlus;
import ij.Prefs;
import ij.gui.Roi;
import ij.process.FloatProcessor;

/**
 * Class to test the use of threads when using Gabor features.
 *
 */
public class TestGaborThreads {
	
	static private final boolean test1() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);
		FeatureStack fs = new FeatureStack(original);
		
		ExecutorService exec = Executors.newFixedThreadPool(8);
		try {
			fs.getGabor(original, 4, 1.0, 1.0, 1.0, 4, exec);
		} finally {
			exec.shutdown();
		}
		
		return printThreads();
	}
	
	static private final boolean test2() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);
		FeatureStack fs = new FeatureStack(original);

		fs.setEnabledFeature("Gabor", true);
		fs.updateFeaturesMT( Prefs.getThreads() );
		fs.shutDownNow();

		printThreads();
		try {
			// Give it time to remove the threads
			Thread.sleep(500);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return printThreads();
	}

	static private final boolean test3() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);

		FeatureStackArray featuresArray = new FeatureStackArray(original.getStackSize());

		// Selected attributes (image features)
		boolean[] enableFeatures = new boolean[]{
				false,   /* Gaussian_blur */
				false,   /* Sobel_filter */
				false,   /* Hessian */
				false,   /* Difference_of_gaussians */
				false,   /* Membrane_projections */
				false,  /* Variance */
				false,  /* Mean */
				false,  /* Minimum */
				false,  /* Maximum */
				false,  /* Median */
				false,  /* Anisotropic_diffusion */
				false,  /* Bilateral */
				false,  /* Lipschitz */
				false,  /* Kuwahara */
				true,  /* Gabor */
				false,  /* Derivatives */
				false,  /* Laplacian */
				false,  /* Structure */
				false,  /* Entropy */
				false   /* Neighbors */
		};
		featuresArray.setEnabledFeatures(enableFeatures);
		featuresArray.updateFeaturesMT();
		featuresArray.shutDownNow();

		printThreads();
		try {
			// Give it time to remove the threads
			Thread.sleep(500);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return printThreads();
	}
	static private final boolean test4() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);
		
		WekaSegmentation seg = new WekaSegmentation( original );

		// Selected attributes (image features)
		boolean[] enableFeatures = new boolean[]{
				false,   /* Gaussian_blur */
				false,   /* Sobel_filter */
				false,   /* Hessian */
				false,   /* Difference_of_gaussians */
				false,   /* Membrane_projections */
				false,  /* Variance */
				false,  /* Mean */
				false,  /* Minimum */
				false,  /* Maximum */
				false,  /* Median */
				false,  /* Anisotropic_diffusion */
				false,  /* Bilateral */
				false,  /* Lipschitz */
				false,  /* Kuwahara */
				true,  /* Gabor */
				false,  /* Derivatives */
				false,  /* Laplacian */
				false,  /* Structure */
				false,  /* Entropy */
				false   /* Neighbors */
		};
		seg.setEnabledFeatures(enableFeatures);
		Roi roi = new Roi( 10, 10, 10, 10 );
		seg.addExample( 0, roi, 1);
		roi = new Roi( 40, 40, 10, 10 );
		seg.addExample( 1, roi, 1);
		seg.trainClassifier();
		//seg.applyClassifier(true);

		printThreads();
		try {
			// Give it time to remove the threads
			Thread.sleep(500);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return printThreads();
	}

	private static boolean printThreads() {
		ThreadGroup rootGroup = Thread.currentThread().getThreadGroup();
		ThreadGroup parentGroup;
		while ((parentGroup = rootGroup.getParent()) != null) {
		    rootGroup = parentGroup;
		}
		Thread[] threads = new Thread[rootGroup.activeCount()];
		while (rootGroup.enumerate(threads, true ) == threads.length) {
		    threads = new Thread[threads.length * 2];
		}
		int count = 0;
		int alive = 0;
		for (int i=0; i<threads.length; ++i) {
			if (null == threads[i]) break;
			++count;
			if (threads[i].isAlive()) ++alive;
		}
		System.out.println("Number of Thread instances found: " + count + " of which alive: " + alive);
		return count == alive;
	}
	
	/**
	 * Method with all possible tests to detect any error when using Gabor filters.
	 */
	@Test
	public void test() {
		assert test1();
		assert test2();
		assert test3();
		assert test4();
	}
}
