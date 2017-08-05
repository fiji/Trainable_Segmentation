package trainableSegmentation;

import ij.IJ;
import trainableSegmentation.unsupervised.Color_Clustering;
/**
 * Class for testing the Color Clustering plugin
 * 
 */
public class TestColorClustering {

    /**
     * Main method for testing Color Clustering plugin
     * @param args
     */
    public static void main(String[] args){
        Class<?> clazz = Color_Clustering.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
        System.setProperty("plugins.dir", pluginsDir);
        IJ.runPlugIn(clazz.getName(),"");

    }
}
