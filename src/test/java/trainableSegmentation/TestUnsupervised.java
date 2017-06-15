package trainableSegmentation;
import ij.IJ;
import ij.ImagePlus;
import trainableSegmentation.unsupervised.ColorClustering;
import trainableSegmentation.unsupervised.PixelClustering;

public class TestUnsupervised {

    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        ColorClustering colorClustering = new ColorClustering(image,30);
        PixelClustering example = new PixelClustering(colorClustering.getFeaturesInstances(),3);
        colorClustering.createFile("example.arff");
        image.show();

    }


}
