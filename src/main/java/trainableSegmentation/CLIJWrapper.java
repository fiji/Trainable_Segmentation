package trainableSegmentation;

import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.kernels.Kernels;

import java.util.HashMap;

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
    private static CLIJ clij = null;

    private static Object lock = new Object();

    public static ImagePlus computeGaussianBlur(final double sigma,
                                              final ImagePlus imp)
    {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJ " + clij);
            clij.op().blur(clijInput, clijOutput, (float) sigma, (float) sigma);

            ImagePlus result = clij.pull(clijOutput);
            return result;
        }
    }


    public static ImagePlus computeDoG(float sigma1, float sigma2, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJ " + clij);

            ClearCLBuffer temp1 = clij.create(clijOutput);
            ClearCLBuffer temp2 = clij.create(clijOutput);

            float magic_number = 0.4f;

            clij.op().blur(clijInput, temp1, magic_number * sigma1, magic_number * sigma1);
            clij.op().blur(clijInput, temp2, magic_number * sigma2, magic_number * sigma2);
            clij.op().subtractImages(temp2, temp1, clijOutput);

            temp1.close();
            temp2.close();

            ImagePlus result = clij.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeMean(float radius, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJ " + clij);
            clij.op().meanBox(clijInput, clijOutput, (int)radius, (int)radius, 0);

            ImagePlus result = clij.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeMin(float radius, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJ " + clij);
            clij.op().minimumBox(clijInput, clijOutput, (int)radius, (int)radius, 0);

            ImagePlus result = clij.pull(clijOutput);
            return result;
        }
    }

    public static ImagePlus computeEntropie(int radius, int numBins, ImagePlus imp) {
        synchronized (lock) { // supress parallelisation here; the GPU does it parallel anyway and we can reuse memory
            checkCache(imp);
            System.out.println("CLIJ " + clij);

            HashMap<String, Object> parameters = new HashMap();
            parameters.put("src", clijInput);
            parameters.put("dst", clijOutput);
            parameters.put("radius", radius);
            parameters.put("numBins", numBins);

            clij.execute( "entropie.cl", "entropie", parameters);

            ImagePlus result = clij.pull(clijOutput);
            return result;
        }
    }


    private static void checkCache(ImagePlus imp) {
        if (imp != cachedImagePlus) {
            clearCache();
            cachedImagePlus = imp;
            clij = CLIJ.getInstance();
            IJ.log(clij.getGPUName());
            clijInput = clij.push(imp);
            clijOutput = clij.create(clijInput.getDimensions(), NativeTypeEnum.Float);
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
        clij = null;
    }


}
