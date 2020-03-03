package trainableSegmentation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import org.junit.Test;

import java.util.NoSuchElementException;

import static org.junit.Assert.assertEquals;

public class FeatureStackTest {

	@Test
	public void testHessian() {
		ImagePlus input = createTestImage();
		ImageStack stack = calculateSingleFeature(input, "Hessian");
		float eigenvalue1 = getCenterPixel(getProcessorBySliceLabel(stack, "Hessian_Eigenvalue_1_0.0"));
		float eigenvalue2 = getCenterPixel(getProcessorBySliceLabel(stack, "Hessian_Eigenvalue_2_0.0"));
		double expectedEigenvalue1 = Math.sqrt(5) + 2; // first eigenvalue of the matrix [1 2; 2 3]
		double expectedEigenvalue2 = -Math.sqrt(5) + 2; // second eigenvalue of the matrix [1 2; 2 3]
		int factor = 64;
		assertEquals(expectedEigenvalue1, eigenvalue1 / factor, 0.0001);
		assertEquals(expectedEigenvalue2, eigenvalue2 / factor, 0.0001);
	}

	/**
	 * Returns an image, whose hessian matrix is equal to:
	 * [1 2]
	 * [2 3]
	 * for every pixel.
	 */
	private ImagePlus createTestImage() {
		ImagePlus input = IJ.createImage("input", 8, 8, 1, 32);
		ImageProcessor processor = input.getProcessor();
		for (int y = 0; y < processor.getHeight(); y++)
			for (int x = 0; x < processor.getHeight(); x++) {
				processor.setf(x, y, 0.5f * 1 * x * x + 2 * x * y + 0.5f * 3 * y * y);
			}
		return input;
	}

	private ImageStack calculateSingleFeature(ImagePlus input, String featureName) {
		FeatureStack featureStack = new FeatureStack(input);
		featureStack.setMinimumSigma(1);
		featureStack.setMaximumSigma(1);
		featureStack.setEnabledFeatures(new boolean[20]);
		featureStack.setEnabledFeature(featureName, true);
		featureStack.updateFeaturesST();
		return featureStack.getStack();
	}

	private ImageProcessor getProcessorBySliceLabel(ImageStack stack, String sliceLabel) {
		for (int i = 1; i <= stack.getSize(); i++) {
			if (sliceLabel.equals(stack.getSliceLabel(i)))
				return stack.getProcessor(i);
		}
		throw new NoSuchElementException();
	}

	private float getCenterPixel(ImageProcessor imageProcessor) {
		return imageProcessor.getf(imageProcessor.getWidth() / 2, imageProcessor.getHeight() / 2);
	}

}
