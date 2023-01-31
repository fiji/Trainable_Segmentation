/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2023 Fiji developers.
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
package trainableSegmentation.unsupervised;

import ij.IJ;
import weka.clusterers.AbstractClusterer;
import weka.core.Instances;

/**
 * This class contains all the library methods to perform
 * unsupervised learning on pixels based on the Weka clusterers.
 * @author Josu Salinas and Ignacio Arganda-Carreras
 *
 */
public class PixelClustering {

    /** Set of instances used to create the clusterer */
    private Instances featuresInstances;
    /** Clustering Weka model */
    private AbstractClusterer selectedClusterer;



    /**
     * Empty constructor
     */
    public PixelClustering(){

    }

    /**
     * Construct PixelClustering object with a specific set of instances
     * and clustering model
     * @param featuresInstances set of instances to use for building the clusterer
     * @param clusterer clustering Weka model to be used
     */
    public PixelClustering(Instances featuresInstances, AbstractClusterer clusterer){
        this.setFeaturesInstances(featuresInstances);
        this.setSelectedClusterer(clusterer);
    }

    /**
     * Build current clusterer based on the current instances
     */
    public void buildClusterer(){
        try {
            selectedClusterer.buildClusterer(featuresInstances);
        } catch (InterruptedException ie) {
            IJ.log("Clusterer building was interrupted");
        } catch (Exception e) {
            IJ.log(e.getMessage());
            e.printStackTrace();
        }
    }


    //Getters and setters

    /**
     * Set feature instances to be used
     * @param featuresInstances instances to be used
     */
    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }

    /**
     * Get features instances that are being used
     * @return instances that are being used
     */
    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    /**
     * Get selected clusterer
     * @return current clusterer
     */
    public AbstractClusterer getSelectedClusterer() {
        return selectedClusterer;
    }

    /**
     * Set selected clusterer
     * @param selectedClusterer clustering Weka model
     */
    public void setSelectedClusterer(AbstractClusterer selectedClusterer) {
        this.selectedClusterer = selectedClusterer;
    }

}
