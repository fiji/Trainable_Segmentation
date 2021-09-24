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
import ij.IJ;
import ij.ImagePlus;
import trainableSegmentation.unsupervised.ColorClustering;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.Canopy;

import java.util.ArrayList;

/**
 * Test class for unsupervised learning in color images
 *
 */
public class TestUnsupervised {

    /**
     * Main method of test class
     *
     * @param args main arguments (not used)
     */
    public static void main( final String[] args )
    {
        ImagePlus image = IJ.openImage();
        image.show();
        ArrayList<ColorClustering.Channel> channels = new ArrayList<ColorClustering.Channel>();
        channels.add(ColorClustering.Channel.fromLabel("Lightness"));
        channels.add(ColorClustering.Channel.fromLabel("a"));
        channels.add(ColorClustering.Channel.fromLabel("b"));
        channels.add(ColorClustering.Channel.fromLabel("Red"));
        channels.add(ColorClustering.Channel.fromLabel("Green"));
        channels.add(ColorClustering.Channel.fromLabel("Blue"));
        channels.add(ColorClustering.Channel.fromLabel("Hue"));
        channels.add(ColorClustering.Channel.fromLabel("Brightness"));
        channels.add(ColorClustering.Channel.fromLabel("Saturation"));
        ColorClustering colorClustering = new ColorClustering(image,14, channels);
        Canopy clusterer = new Canopy();
        try {
            clusterer.setNumClusters(2);
        } catch (Exception e) {
            e.printStackTrace();
        }
        AbstractClusterer theClusterer = colorClustering.createClusterer(clusterer);
        colorClustering.setTheClusterer(theClusterer);
        //colorClustering.createFile(colorClustering.getFeaturesInstances());
        FeatureStackArray theFeatures = colorClustering.createFSArray(image);
        ImagePlus probMap = colorClustering.createProbabilityMaps(colorClustering.getFeatureStackArray());
        probMap.show();
        ImagePlus clusteredImage = colorClustering.createClusteredImage(theFeatures);
        clusteredImage.show();
    }



}

