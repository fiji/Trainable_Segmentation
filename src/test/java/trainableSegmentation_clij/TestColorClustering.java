package trainableSegmentation_clij;

import fiji.Debug;

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
    	// Call the plugin with empty arguments (this will pop up an Open dialog)
    	Debug.run( "Color Clustering", "" );
    }
}
