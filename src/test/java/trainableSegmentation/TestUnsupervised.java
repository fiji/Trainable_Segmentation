package trainableSegmentation;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.NewImage;
import ij.process.ImageProcessor;
import trainableSegmentation.unsupervised.ColorClustering;
import trainableSegmentation.unsupervised.PixelClustering;
import weka.clusterers.Clusterer;
import weka.core.Instances;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class TestUnsupervised {

    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        image.show();
        ColorClustering colorClustering = new ColorClustering(image,30,2);
        colorClustering.createFile("test.arff");
        ImagePlus clusteredImage = colorClustering.createClusteredImage();
        clusteredImage.show();
    }



}
