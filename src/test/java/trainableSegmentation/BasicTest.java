package trainableSegmentation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;
import hr.irb.fastRandomForest.FastRandomForest;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.process.ByteProcessor;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;

import java.net.URL;
import java.util.function.Consumer;

import org.junit.Ignore;
import org.junit.Test;

public class BasicTest
{
	@Test
	public void test1()
	{
		final ImagePlus image = makeTestImage( "test", 2, 2, 17, 2, 123, 54 );
		final ImagePlus labels = makeTestImage( "labels", 2, 2, 1, 1, 0, 0 );

		WekaSegmentation segmentator = new WekaSegmentation( image );

		if ( false == segmentator.addBinaryData(image, labels, "class 2", "class 1") )
			assertTrue("Error while adding binary data to segmentator", false);


		assertTrue("Failed to train classifier", true == segmentator.trainClassifier());

		segmentator.applyClassifier( false );

		ImagePlus result = segmentator.getClassifiedImage();


		assertTrue("Failed to apply trained classifier", null != result);

		byte[] pix = (byte[]) result.getProcessor().getPixels();
		byte[] pixTrue = (byte[]) labels.getProcessor().getPixels();
		for( int i=0; i<pix.length; i++)
		{
			assertTrue("Misclassified training sample", pix[i] == pixTrue[i]);
		}
	}

	@Test
	public void bridge() {
		final ImagePlus bridge = loadFromResource( "/bridge.png" );
		assumeNotNull( bridge );
		final ImagePlus bridgeExpect = loadFromResource( "/bridge-expected.png" );
		assumeNotNull( bridgeExpect );

		ImagePlus output = segmentBridge(bridge);
		assertEquals(0, diffImagePlus(output, bridgeExpect));
	}

	@Test
	public void testDefaultFeatureGenerationST() {
		testDefaultFeaturesOnBridge(FeatureStack::updateFeaturesST);
	}

	@Test
	public void testDefaultFeatureGenerationMT() {
		testDefaultFeaturesOnBridge(FeatureStack::updateFeaturesMT);
	}

	private void testDefaultFeaturesOnBridge(Consumer<FeatureStack> updateFeaturesMethod) {
		// setup
		final ImagePlus bridge = loadFromResource("/nuclei.tif");

		// process
		final FeatureStack featureStack = new FeatureStack(bridge);
		updateFeaturesMethod.accept(featureStack);
		final ImagePlus features = new ImagePlus("features", featureStack.getStack());
		// test
		final ImagePlus expected = loadFromResource("/nuclei-features.tif");
		assertEquals(0, diffImagePlus(expected, features));
	}

	private ImagePlus makeTestImage(final String title, final int width, final int height, final int... pixels)
	{
		assertEquals( pixels.length, width * height );
		final byte[] bytes = new byte[pixels.length];
		for (int i = 0; i < bytes.length; i++) bytes[i] = (byte)pixels[i];
		final ByteProcessor bp = new ByteProcessor( width, height, bytes, null );
		return new ImagePlus( title, bp );
	}

	private static ImagePlus segmentBridge(final ImagePlus bridge) {
		WekaSegmentation segmentator = new WekaSegmentation( bridge );
		segmentator.addExample( 0, new Roi( 10, 10, 50, 50 ), 1 );
		segmentator.addExample( 1, new Roi( 400, 400, 30, 30 ), 1 );

		FastRandomForest rf = (FastRandomForest) segmentator.getClassifier();
		rf.setSeed( 69 );
		assertTrue( segmentator.trainClassifier() );

		segmentator.applyClassifier( false );
		return segmentator.getClassifiedImage();
	}

	private ImagePlus loadFromResource(final String path) {
		final URL url = getClass().getResource(path);
		if (url == null) return null;
		if ("file".equals(url.getProtocol())) return new ImagePlus(url.getPath());
		return new ImagePlus(url.toString());
	}

	private int diffImagePlus(final ImagePlus a, final ImagePlus b) {
		final int[] dimsA = a.getDimensions(), dimsB = b.getDimensions();
		if (dimsA.length != dimsB.length) return dimsA.length - dimsB.length;
		for (int i = 0; i < dimsA.length; i++) {
			if (dimsA[i] != dimsB[i]) return dimsA[i] - dimsB[i];
		}
		int count = 0;
		final ImageStack stackA = a.getStack(), stackB = b.getStack();
		for (int slice = 1; slice <= stackA.getSize(); slice++) {
			count += diff( stackA.getProcessor( slice ), stackB.getProcessor( slice ) );
		}
		return count;
	}

	private int diff(final ImageProcessor a, final ImageProcessor b) {
		int count = 0;
		final int width = a.getWidth(), height = a.getHeight();
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (a.getf(x, y) != b.getf(x, y)) count++;
			}
		}
		return count;
	}

	public static void main(String...strings) {
		final String pathOfClass = "/" + BasicTest.class.getName().replace('.', '/') + ".class";
		final URL url = BasicTest.class.getResource(pathOfClass);
		if ( !"file".equals( url.getProtocol() ) ) throw new RuntimeException( "Need to run from test-classes/" );
		final String suffix = "/target/test-classes" + pathOfClass;
		final String path = url.getPath();
		if ( !path.endsWith( suffix ) ) throw new RuntimeException( "Unexpected class location: " + path );
		final String resources = path.substring( 0, path.length() - suffix.length() ) + "/src/test/resources/";
		final ImagePlus bridge = new ImagePlus( "http://imagej.nih.gov/ij/images/bridge.gif" );
		IJ.save( bridge, resources + "bridge.png" );
		final ImagePlus bridgeExpected = segmentBridge( bridge );
		IJ.save( bridgeExpected, resources + "bridge-expected.png" );
	}

}
