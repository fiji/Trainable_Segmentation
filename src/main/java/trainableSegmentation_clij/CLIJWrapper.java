package trainableSegmentation_clij;

import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clijx.CLIJx;

/**
 * Analogously to ImageScience, we use CLIJ to compute feature images
 *
 * Todo: Consider turning this class of static methods into a singleton
 *
 * @Author: Robert Haase, rhaase@mpi-cbg.de
 * July 2019
 */
public class CLIJWrapper {
    private static ImagePlus cachedImagePlus = null;
    private static ClearCLBuffer clijInput = null;
    private static ClearCLBuffer clijOutput = null;
    private static CLIJx clijx = null;

    private static Object lock = new Object();

    public static ImagePlus computeGaussianBlur(final double sigma,
                                              final ImagePlus imp)
    {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);
            clijx.gaussianBlur(clijInput, clijOutput, (float) sigma, (float) sigma);

            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }


    public static ImagePlus computeDoG(float sigma1, float sigma2, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);

            ClearCLBuffer temp1 = clijx.create(clijOutput);
            ClearCLBuffer temp2 = clijx.create(clijOutput);

            float magic_number = 0.4f;

            clijx.gaussianBlur(clijInput, temp1, magic_number * sigma1, magic_number * sigma1);
            clijx.gaussianBlur(clijInput, temp2, magic_number * sigma2, magic_number * sigma2);
            clijx.subtractImages(temp2, temp1, clijOutput);

            clijx.release(temp1);
            clijx.release(temp2);

            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeMean(float radius, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);
            clijx.meanBox(clijInput, clijOutput, (int)radius, (int)radius, 0);

            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeMin(float radius, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);
            clijx.minimumBox(clijInput, clijOutput, (int)radius, (int)radius, 0);

            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeMax(float radius, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);
            clijx.maximumBox(clijInput, clijOutput, (int)radius, (int)radius, 0);

            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeEntropy(int radius, int numBins, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJx " + clijx);
/*
            HashMap<String, Object> parameters = new HashMap();
            parameters.put("src", clijInput);
            parameters.put("dst", clijOutput);
            parameters.put("radius", radius);
            parameters.put("numBins", numBins);

            clij.execute( "entropie.cl", "entropie", parameters);
*/
            clijx.entropyBox(clijInput, clijOutput, radius, radius, 0);
            ImagePlus result = clijx.pull(clijOutput);
            return result;
        }
    }


    private static void checkCache(ImagePlus imp) {
        if (imp != cachedImagePlus) {
            clearCache();
            cachedImagePlus = imp;
            clijx = CLIJx.getInstance();
            clijInput = clijx.push(imp);
            clijOutput = clijx.create(clijInput.getDimensions(), NativeTypeEnum.Float);
        }
    }

    public static void clearCache() {
        cachedImagePlus = null;
        if (clijInput != null) {
            clijInput.close();
        }
        if (clijOutput != null) {
            clijOutput.close();
        }
        clijx = null;
    }
}
