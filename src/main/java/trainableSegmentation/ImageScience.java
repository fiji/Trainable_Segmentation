/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2021 Fiji developers.
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

import ij.ImagePlus;
import imagescience.feature.Differentiator;
import imagescience.feature.Edges;
import imagescience.feature.Hessian;
import imagescience.feature.Laplacian;
import imagescience.feature.Structure;
import imagescience.image.Aspects;
import imagescience.image.FloatImage;
import imagescience.image.Image;

import java.util.ArrayList;
import java.util.Vector;

/**
 * Helper class which encapsulates access to the optional ImageScience library.
 */
public final class ImageScience {

	private ImageScience() {
		// prevent instantiation of utility class
	}

	/**
	 * Tests whether the ImageScience library is installed.
	 * 
	 * @return true if ImageScience classes are available.
	 * @throws NoClassDefFoundError if ImageScience classes cannot be loaded.
	 */
	public static boolean isAvailable() {
		Image.class.getName();
		return true;
	}

	public static ArrayList<ImagePlus> computeEigenimages(final double sigma,
		final double integrationScale, final ImagePlus imp)
	{
		final Image img = Image.wrap(imp);
		final Aspects aspects = img.aspects();
		final Image newimg = new FloatImage(img);

		final Structure structure = new Structure();
		final Vector<Image> eigenimages = structure.run(newimg, sigma, integrationScale);

		final int nrimgs = eigenimages.size();
		for (int i=0; i<nrimgs; ++i)
			eigenimages.get(i).aspects(aspects);

		final ArrayList<ImagePlus> result = new ArrayList<ImagePlus>(nrimgs);
		for (final Image eigenimage : eigenimages)
			result.add(eigenimage.imageplus());
		return result;
	}

	public static ImagePlus computeDerivativeImage(final double sigma,
		final int xOrder, final int yOrder, ImagePlus imp)
	{
		final Image img = Image.wrap(imp);
		final Aspects aspects = img.aspects();
		final Image newimg = new FloatImage(img);

		final Differentiator diff = new Differentiator();

		diff.run(newimg, sigma , xOrder, yOrder, 0);
		newimg.aspects(aspects);

		return newimg.imageplus();
	}

	public static ImagePlus computeLaplacianImage(final double sigma,
		final ImagePlus imp)
	{
		final Image img = Image.wrap(imp) ;
		final Aspects aspects = img.aspects();
		Image newimg = new FloatImage(img);

		final Laplacian laplace = new Laplacian();

		newimg = laplace.run(newimg, sigma);
		newimg.aspects(aspects);

		return newimg.imageplus();
	}

	public static ImagePlus computeDifferentialImage(final double sigma,
		final int xOrder, final int yOrder, final int zOrder,
		final ImagePlus imp)
	{
		Image img = Image.wrap(imp);
		Aspects aspects = img.aspects();

		Image newimg = new FloatImage(img);
		Differentiator diff = new Differentiator();

		diff.run(newimg, sigma , xOrder, yOrder, zOrder);
		newimg.aspects(aspects);

		final ImagePlus ip = newimg.imageplus();
		return ip;
	}

	public static ArrayList<ImagePlus> computeHessianImages(final double sigma,
		final boolean absolute, final ImagePlus imp)
	{
		final Image img = Image.wrap(imp);
		final Aspects aspects = img.aspects();

		final Image newimg = new FloatImage(img);
		final Hessian hessian = new Hessian();

		final Vector<Image> hessianImages = hessian.run(newimg, sigma, absolute);

		final int nrimgs = hessianImages.size();
		for (int i=0; i<nrimgs; ++i)
			hessianImages.get(i).aspects(aspects);

		final ArrayList<ImagePlus> result = new ArrayList<ImagePlus>(nrimgs);
		for (final Image hessianImage : hessianImages)
			result.add(hessianImage.imageplus());
		return result;
	}

	public static ImagePlus computeEdgesImage(final double sigma,
		final ImagePlus imp)
	{
		final Image img = Image.wrap(imp);
		final Aspects aspects = img.aspects();

		Image newimg = new FloatImage(img);
		final Edges edges = new Edges();

		newimg = edges.run(newimg, sigma, false);
		newimg.aspects(aspects);

		return newimg.imageplus();
	}

}
