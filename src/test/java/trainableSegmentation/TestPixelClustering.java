package trainableSegmentation;
import ij.IJ;
import ij.ImagePlus;
import trainableSegmentation.unsupervised.PixelClustering;

public class TestPixelClustering {

    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        PixelClustering example = new PixelClustering(image);

    }


}
