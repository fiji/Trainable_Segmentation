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
