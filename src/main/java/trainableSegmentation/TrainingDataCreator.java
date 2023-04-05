package trainableSegmentation;

import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.process.FloatPolygon;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;

public class TrainingDataCreator {
    public Instances createTrainingInstances(FeatureStack featureStack, ArrayList<Roi>[] examples, int numOfClasses, String[] classLabels) {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 1; i <= featureStack.getSize(); i++) {
            String attString = featureStack.getSliceLabel(i);
            attributes.add(new Attribute(attString));
        }

        final ArrayList<String> classes = new ArrayList<String>();

        int numOfInstances = 0;
        for (int i = 0; i < numOfClasses; i++) {
            // Do not add empty lists
            if (!examples[i].isEmpty())
                classes.add(classLabels[i]);
            numOfInstances += examples[i].size();
        }

        attributes.add(new Attribute("class", classes));

        final Instances trainingData = new Instances("segment", attributes, numOfInstances);

        // For all classes
        for (int l = 0; l < numOfClasses; l++) {
            int nl = 0;
            // Read all lists of examples
            for (int j = 0; j < examples[l].size(); j++) {
                Roi r = examples[l].get(j);


                // For polygon rois we get the list of points
                if (r instanceof PolygonRoi && r.getType() != Roi.FREEROI) {
                    if (r.getStrokeWidth() == 1) {
                        int[] x = r.getPolygon().xpoints;
                        int[] y = r.getPolygon().ypoints;
                        final int n = r.getPolygon().npoints;

                        for (int i = 0; i < n; i++) {
                            double[] values = new double[featureStack.getSize() + 1];
                            for (int z = 1; z <= featureStack.getSize(); z++)
                                values[z - 1] = featureStack.getProcessor(z).getPixelValue(x[i], y[i]);
                            values[featureStack.getSize()] = (double) l;
                            trainingData.add(new DenseInstance(1.0, values));
                            // increase number of instances for this class
                            nl++;
                        }
                    } else // For thicker lines, include also neighbors
                    {
                        final int width = (int) Math.round(r.getStrokeWidth());
                        FloatPolygon p = r.getFloatPolygon();
                        int n = p.npoints;

                        double x1, y1;
                        double x2 = p.xpoints[0] - (p.xpoints[1] - p.xpoints[0]);
                        double y2 = p.ypoints[0] - (p.ypoints[1] - p.ypoints[0]);
                        for (int i = 0; i < n; i++) {
                            x1 = x2;
                            y1 = y2;
                            x2 = p.xpoints[i];
                            y2 = p.ypoints[i];

                            double dx = x2 - x1;
                            double dy = y2 - y1;
                            double length = Math.sqrt(dx*dx + dy*dy);
                            dx /= length;
                            dy /= length;
                            // Add feature vectors for all pixels along the line segment
                            for (int w = 0; w < width; w++) {
                                double[] values = new double[featureStack.getSize() + 1];
                                double xOffset = (w - (width - 1) / 2.0) * dy;
                                double yOffset = (w - (width - 1) / 2.0) * dx;

                                for (int z = 1; z <= featureStack.getSize(); z++)
                                    values[z - 1] = featureStack.getProcessor(z).getInterpolatedValue(x1 + xOffset, y1 + yOffset);

                                values[featureStack.getSize()] = (double) l;
                                trainingData.add(new DenseInstance(1.0, values));
                                // increase number of instances for this class
                                nl++;
                            }
                        }
                    }
                }
            }
            System.out.println(nl + " instances added to class " + classLabels[l]);
        }

        // set class index to last attribute
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        return trainingData;
    }
}