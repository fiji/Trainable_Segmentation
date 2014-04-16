package trainableSegmentation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import ij.ImagePlus;
import ij.process.ByteProcessor;

import org.junit.Test;

public class BasicTest
{
	@Test
	public void test1()
	{
		final ImagePlus image = makeTestImage( "test", 2, 2, 17, 2, 123, 54 );
		final ImagePlus labels = makeTestImage( "labels", 2, 2, 255, 255, 0, 0 );

		WekaSegmentation segmentator = new WekaSegmentation( image );

		if ( false == segmentator.addBinaryData(image, labels, "class 2", "class 1") )
			assertTrue("Error while adding binary data to segmentator", false);


		assertTrue("Failed to train classifier", true == segmentator.trainClassifier());

		segmentator.applyClassifier( false );

		ImagePlus result = segmentator.getClassifiedImage();


		assertTrue("Failed to apply trained classifier", null != result);

		float[] pix = (float[]) result.getProcessor().getPixels();
		byte[] pixTrue = (byte[]) labels.getProcessor().getPixels();
		for( int i=0; i<pix.length; i++)
		{
			assertTrue("Misclassified training sample", pix[i] * 255 == (pixTrue[i]&0xff) );
		}
	}

	private ImagePlus makeTestImage(final String title, final int width, final int height, final int... pixels)
	{
		assertEquals( pixels.length, width * height );
		final byte[] bytes = new byte[pixels.length];
		for (int i = 0; i < bytes.length; i++) bytes[i] = (byte)pixels[i];
		final ByteProcessor bp = new ByteProcessor( width, height, bytes, null );
		return new ImagePlus( title, bp );
	}

}
