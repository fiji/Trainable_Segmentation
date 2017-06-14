package trainableSegmentation.unsupervised;
/*
TrainableSegmentation test case: TestPixelClustering
 */




import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Line;
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.gui.ShapeRoi;
import ij.process.ByteProcessor;
import ij.process.ColorSpaceConverter;
import ij.process.FloatPolygon;
import ij.process.ImageProcessor;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import trainableSegmentation.WekaSegmentation;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

public class PixelClustering {

    private ImagePlus image;

    private AbstractClusterer clusterer;
    private FeatureStackArray featureStackArray;
    private Instances featuresInstances;
   /* private float minimumSigma = 1f;
    private float maximumSigma = 16f;
    private boolean useNeighbors = false;
    private int membraneThickness = 1;
    private int membranePatchSize = 19;
    private boolean[] enabledFeatures = new boolean[]{*/
           /* true, 	 /*Gaussian_blur */
            /*true, 	/* Sobel_filter */
            /*true, 	/* Hessian */
            /*true, 	/* Difference_of_gaussians */
            /*true, 	/* Membrane_projections */
            /*false, 	/* Variance */
            /*false, 	/* Mean */
            /*false, 	/* Minimum */
            /*false, 	/* Maximum */
            /*false, 	/* Median */
            /*false,	/* Anisotropic_diffusion */
            /*false, 	/* Bilateral */
            /*false, 	/* Lipschitz */
            /*alse, 	/* Kuwahara */
           /* false,	/* Gabor */
            /*false, 	/* Derivatives */
            /*false, 	/* Laplacian */
            /*false,	/* Structure */
            /*false,	/* Entropy */
            /*false	/* Neighbors */
     /*};*/



    public void setFeaturesInstances(Instances featuresInstances) {
        this.featuresInstances = featuresInstances;
    }

    public Instances getFeaturesInstances() {
        return featuresInstances;
    }

    public PixelClustering(){

        IJ.log("Created PixelClustering object with no variable declaration"); //Delete

    }


    public PixelClustering(ImagePlus imagePlus){//create feature stack array and save it into an arff file to check


        this.image = imagePlus;
        ImageProcessor imageProcessor = image.getProcessor();
        int numClusters = 3;//Number of clusters
        int numOfClasses = 3;//Number of classes
        int numSamples = 3; //Number of Samples
        String[] classLabels = new String[numOfClasses];
        classLabels[0]="x";classLabels[1]="y";classLabels[2]="z"; //Names for classes
        featureStackArray = new FeatureStackArray(image.getStackSize());
        for(int slice = 1; slice <= image.getStackSize(); ++slice){
            ImageStack stack = new ImageStack(image.getWidth(),image.getHeight());

            ColorSpaceConverter converter = new ColorSpaceConverter();

            ImagePlus lab = converter.RGBToLab(new ImagePlus("RGB",image.getStack().getProcessor(slice)));

            stack.addSlice("a", lab.getStack().getProcessor(2));
            stack.addSlice("b",lab.getStack().getProcessor(3));

            FeatureStack features = new FeatureStack(stack.getWidth(),stack.getHeight(),false);
            features.setStack(stack);

            featureStackArray.set(features,slice-1);
           /* byte[] pixels = new byte[image.getWidth() * image.getHeight()];
            for(int i=0;i<pixels.length;++i){
                pixels[i] = (byte) (i % numClusters + 1);
            }
            ByteProcessor labels = new ByteProcessor(image.getWidth(),image.getHeight(),pixels);*/
            if( null == featuresInstances )
            {
                IJ.log("Initializing loaded data...");
                // Create instances
                ArrayList<Attribute> attributes = new ArrayList<Attribute>();
                for (int i=1; i<=featureStackArray.get(slice-1).getSize(); i++)
                {
                    String attString = featureStackArray.get(slice-1).getSliceLabel(i);
                    attributes.add( new Attribute( attString ) );
                }

                if( featureStackArray.get(slice-1).useNeighborhood() )
                    for (int i=0; i<8; i++)
                    {
                        IJ.log("Adding extra attribute original_neighbor_" + (i+1) + "...");
                        attributes.add( new Attribute( new String( "original_neighbor_" + (i+1) ) ) );
                    }

                // Update list of names of loaded classes
                /*ArrayList<String> loadedClassNames = new ArrayList<String>();
                for(int i = 0; i < numOfClasses ; i ++)
                    loadedClassNames.add(classLabels[i]);

                attributes.add(new Attribute("class", loadedClassNames));*/
                featuresInstances = new Instances("segment", attributes, 1);

                //featuresInstances.setClassIndex(featuresInstances.numAttributes()-1);
            }
            Random rand = new Random();
            for(int i=0;i<numSamples;++i){
                int randx = rand.nextInt((image.getHeight()-0)+1)+0;//(max-min+1)+min
                int randy = rand.nextInt((image.getWidth()-0)+1)+0;//(max-min+1)+min
                //Have to create instance without class, for the time being I will manually create the instance here
                double[] values = new double[2]; //a & b
                float[] a = (float[]) stack.getPixels(1); //Array of values of a
                float[] b = (float[]) stack.getPixels(2); //Array of values of b
                values[0] = a[randx+image.getWidth()*randy];
                values[1] = b[randx+image.getWidth()*randy];
                featuresInstances.add(new DenseInstance(1.0, values));
            }

            /*ArrayList<Point>[] classCoordinates = new ArrayList[ numOfClasses ];
            for(int i = 0; i < numOfClasses ; ++i) {
                classCoordinates[i] = new ArrayList<Point>();
            }
            Random rand = new Random();
            final int width = image.getWidth();
            final int height = image.getHeight();
            for(int y = 0 ; y < height; y++) {
                for (int x = 0; x < width; x++) { //For all pixels choose a class at random
                    int rando = rand.nextInt((numOfClasses-1-0)+1)+0;//(max-min+1)+min
                    classCoordinates[rando].add(new Point(x, y));
                    }
            }
            for( int i=0; i<numSamples; i++ )
            {
                for( int j = 0; j < numOfClasses ; j ++ )
                {
                    if( !classCoordinates[ j ].isEmpty() )
                    {
                        int randomSample = rand.nextInt( classCoordinates[ j ].size() );

                        featuresInstances.add(features.createInstance( classCoordinates[ j ].get( randomSample ).x, classCoordinates[ j ].get( randomSample ).y, j ) );
                    }
                }
            }*/

            /*for(int y = 0 ; y < height; y++) { //<commented>
                for (int x = 0; x < width; x++) {
                    int classIndex = (int) imageProcessor.getf(x, y) - 1;

                    if (classIndex >= 0 && classIndex < numOfClasses)
                        classCoordinates[classIndex].add(new Point(x, y));
                }
            }
            Random rand = new Random();
            for( int i=0; i<numSamples; i++ )
            {
                for( int j = 0; j < numOfClasses ; j ++ )
                {
                    if( !classCoordinates[ j ].isEmpty() )
                    {
                        int randomSample = rand.nextInt( classCoordinates[ j ].size() );

                        featuresInstances.add(features.createInstance( classCoordinates[ j ].get( randomSample ).x,
                                classCoordinates[ j ].get( randomSample ).y, j ) );
                    }
                }
            }*/ //</commented>
        }

         /*String[] classLabels = new String[numOfClasses]; //<commented>
        classLabels[0]="a";classLabels[1]="b";classLabels[2]="c";
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i=1; i<=featureStackArray.getNumOfFeatures(); i++)
        {
            String attString = featureStackArray.getLabel(i);
            attributes.add(new Attribute(attString));
        }
        final ArrayList<String> classes;
        classes = new ArrayList<String>();
        for(int i = 0; i < numOfClasses ; i ++)
        {
            for(int n=0; n<image.getImageStackSize(); n++)
            {
                if(!classes.contains(classLabels[i]))
                    classes.add(classLabels[i]);
            }
        }


        Vector<ArrayList<Roi>>[] examples = new Vector[image.getImageStackSize()];
        for(int i=0; i< image.getImageStackSize(); i++)
        {
            examples[i] = new Vector<ArrayList<Roi>>(numOfClasses);

            for(int j=0; j<numOfClasses; j++)
                examples[i].add(new ArrayList<Roi>());

            // Initialize each feature stack (one per slice)
            featureStackArray.set(new FeatureStack(image.getImageStack().getProcessor(i+1)), i);
        }


        attributes.add(new Attribute("class", classes));
        featuresInstances =  new Instances( "segment", attributes, 1 );
        featuresInstances.setClassIndex(featureStackArray.getNumOfFeatures());
        final boolean colorFeatures = image.getType() == ImagePlus.COLOR_RGB;
        for(int classIndex = 0; classIndex < numOfClasses; classIndex++)
        {
            int nl = 0;
            // Read all lists of examples
            for(int sliceNum = 1; sliceNum <= image.getImageStackSize(); sliceNum ++)
                for(int j=0; j < examples[sliceNum-1].get( classIndex ).size(); j++)
                {
                    Roi r = examples[ sliceNum-1 ].get( classIndex ).get(j);

                    // For polygon rois we get the list of points
                    if( r instanceof PolygonRoi && r.getType() == Roi.FREELINE )
                    {
                        if(r.getStrokeWidth() == 1)
                            nl += addThinFreeLineSamples(featuresInstances, classIndex,
                                    sliceNum, r);

                        else // For thicker lines, include also neighbors
                            nl += addThickFreeLineInstances(featuresInstances,
                                    colorFeatures, classIndex, sliceNum, r);
                    }
                    else if( r instanceof Line)
                    {
                        // Get all coordinates in the line
                        nl += addLineInstances(featuresInstances, colorFeatures, classIndex,
                                sliceNum, r);
                    }
                    // for regular rectangles
                    else if ( r.getType() == Roi.RECTANGLE && r.getCornerDiameter() == 0 )
                        nl += addRectangleRoiInstances( featuresInstances, classIndex, sliceNum, r );
                    else // for the rest of rois we get ALL points inside the roi
                        nl += addShapeRoiInstances( featuresInstances, classIndex, sliceNum, r );
                }

            IJ.log("# of pixels selected as " + classLabels[classIndex] + ": " +nl);
        }*/ //</commented>



        BufferedWriter out = null;
        try{
            out = new BufferedWriter(
                    new OutputStreamWriter(
                            new FileOutputStream( "test.arff" ), StandardCharsets.UTF_8 ) );

            final Instances header = new Instances(featuresInstances, 0);
            out.write(header.toString());

            for(int i = 0; i < featuresInstances.numInstances(); i++)
            {
                out.write(featuresInstances.get(i).toString()+"\n");
            }
        }
        catch(Exception e)
        {
            IJ.log("Error: couldn't write instances into .ARFF file.");
            IJ.showMessage("Exception while saving data as ARFF file");
            e.printStackTrace();
        }
        finally{
            try {
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        IJ.log("Created file?");



    }

    private int addThinFreeLineSamples(final Instances trainingData, int classIndex,
                                       int sliceNum, Roi r)
    {
        int numInstances = 0;
        int[] x = r.getPolygon().xpoints;
        int[] y = r.getPolygon().ypoints;
        final int n = r.getPolygon().npoints;

        for (int i=0; i<n; i++)
        {
            double[] values = new double[featureStackArray.getNumOfFeatures()+1];

            for (int z=1; z<=featureStackArray.getNumOfFeatures(); z++)
                values[z-1] = featureStackArray.get(sliceNum-1).getProcessor(z).getPixelValue(x[i], y[i]);

            values[featureStackArray.getNumOfFeatures()] = classIndex;
            trainingData.add(new DenseInstance(1.0, values));
            // increase number of instances for this class
            numInstances ++;
        }
        return numInstances;
    }
    public void createInstances(){
        IJ.log("Not implemented sry"); //Delete
    }

    public boolean build(){
        this.createInstances();
        try {
            this.clusterer.buildClusterer(this.featuresInstances);
        } catch (Exception e) {
            IJ.log("Clusterer construction was interrupted.");
            return false;
        }
        // Print classifier information
        IJ.log( this.clusterer.toString() );
        return true;
    }


    public boolean apply(){
        this.createInstances();
        int[] results = new int[this.featuresInstances.numInstances()];
        for(int i=1;i<this.featuresInstances.numInstances();++i){
            try {
                results[i] = this.clusterer.clusterInstance(this.featuresInstances.instance(i));
            } catch (Exception e) {
                IJ.log("Clusterer applying was interrupted.");
                return false;
            }
        }

        return true;
    }

    private int addThickFreeLineInstances(final Instances trainingData,
                                          final boolean colorFeatures, int classIndex, int sliceNum, Roi r)
    {
        final int width = Math.round(r.getStrokeWidth());
        FloatPolygon p = r.getFloatPolygon();
        int n = p.npoints;

        int numInstances = 0;

        double x1, y1;
        double x2=p.xpoints[0]-(p.xpoints[1]-p.xpoints[0]);
        double y2=p.ypoints[0]-(p.ypoints[1]-p.ypoints[0]);
        for (int i=0; i<n; i++)
        {
            x1 = x2;
            y1 = y2;
            x2 = p.xpoints[i];
            y2 = p.ypoints[i];

            double dx = x2-x1;
            double dy = y1-y2;
            double length = (float)Math.sqrt(dx*dx+dy*dy);
            dx /= length;
            dy /= length;
            double x = x2-dy*width/2.0;
            double y = y2-dx*width/2.0;

            int n2 = width;
            do {
                if(x >= 0 && x < featureStackArray.get(sliceNum-1).getWidth()
                        && y >= 0 && y <featureStackArray.get(sliceNum-1).getHeight())
                {
                    double[] values = new double[featureStackArray.getNumOfFeatures()+1];
                    if(colorFeatures)
                        for (int z=1; z<=featureStackArray.getNumOfFeatures(); z++)
                            values[z-1] = featureStackArray.get(sliceNum-1).getProcessor(z).getInterpolatedPixel(x, y);
                    else
                        for (int z=1; z<=featureStackArray.getNumOfFeatures(); z++)
                            values[z-1] = featureStackArray.get(sliceNum-1).getProcessor(z).getInterpolatedValue(x, y);
                    values[featureStackArray.getNumOfFeatures()] = classIndex;
                    trainingData.add(new DenseInstance(1.0, values));
                    // increase number of instances for this class
                    numInstances ++;
                }
                x += dy;
                y += dx;
            } while (--n2>0);
        }
        return numInstances;
    }

    private int addLineInstances(
            final Instances trainingData,
            final boolean colorFeatures,
            int classIndex,
            int sliceNum,
            Roi r)
    {
        int numInstances = 0;
        double dx = ((Line)r).x2d - ((Line)r).x1d;
        double dy = ((Line)r).y2d - ((Line)r).y1d;
        int n = (int) Math.round( Math.sqrt( dx*dx + dy*dy ) );
        double xinc = dx/n;
        double yinc = dy/n;

        double x = ((Line)r).x1d;
        double y = ((Line)r).y1d;

        for (int i=0; i<n; i++)
        {
            if(x >= 0 && x < featureStackArray.get(sliceNum-1).getWidth()
                    && y >= 0 && y <featureStackArray.get(sliceNum-1).getHeight())
            {
                double[] values = new double[featureStackArray.getNumOfFeatures()+1];
                if( colorFeatures )
                    for (int z=1; z<=featureStackArray.getNumOfFeatures(); z++)
                        values[z-1] = featureStackArray.get(sliceNum-1).getProcessor(z).getInterpolatedPixel(x, y);
                else
                    for (int z=1; z<=featureStackArray.getNumOfFeatures(); z++)
                        values[z-1] = featureStackArray.get(sliceNum-1).getProcessor(z).getInterpolatedValue(x, y);
                values[featureStackArray.getNumOfFeatures()] = classIndex;
                trainingData.add(new DenseInstance(1.0, values));
                // increase number of instances for this class
                numInstances ++;
            }

            x += xinc;
            y += yinc;
        }
        return numInstances;
    }

    private int addRectangleRoiInstances(
            final Instances trainingData,
            int classIndex,
            int sliceNum,
            Roi r)
    {
        int numInstances = 0;

        final Rectangle rect = r.getBounds();

        final int x0 = rect.x;
        final int y0 = rect.y;

        final int lastX = x0 + rect.width;
        final int lastY = y0 + rect.height;

        final FeatureStack fs = featureStackArray.get( sliceNum - 1 );

        for( int x = x0; x < lastX; x++ )
            for( int y = y0; y < lastY; y++ )
            {
                trainingData.add( fs.createInstance(x, y, classIndex) );

                // increase number of instances for this class
                numInstances ++;
            }
        return numInstances;
    }

    private int addShapeRoiInstances(
            final Instances trainingData,
            int classIndex,
            int sliceNum,
            Roi r)
    {
        int numInstances = 0;
        final ShapeRoi shapeRoi = new ShapeRoi(r);
        final Rectangle rect = shapeRoi.getBounds();

        int lastX = rect.x + rect.width ;
        if( lastX >= image.getWidth() )
            lastX = image.getWidth() - 1;
        int lastY = rect.y + rect.height;
        if( lastY >= image.getHeight() )
            lastY = image.getHeight() - 1;
        int firstX = Math.max( rect.x, 0 );
        int firstY = Math.max( rect.y, 0 );

        final FeatureStack fs = featureStackArray.get( sliceNum - 1 );

        // create equivalent binary image to speed up the checking
        // of each pixel belonging to the shape
        final ByteProcessor bp = new ByteProcessor( rect.width, rect.height );
        bp.setValue( 255 );
        shapeRoi.setLocation( 0 , 0 );
        bp.fill( shapeRoi );

        for( int x = firstX, rectX = 0; x < lastX; x++, rectX++ )
            for( int y = firstY, rectY = 0; y < lastY; y++, rectY++ )
                if( bp.getf(rectX, rectY) > 0 )
                {
                    trainingData.add( fs.createInstance( x, y, classIndex ) );

                    // increase number of instances for this class
                    numInstances ++;
                }
        return numInstances;
    }





}
