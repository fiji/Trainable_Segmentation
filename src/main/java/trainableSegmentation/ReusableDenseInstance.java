package trainableSegmentation;

import weka.core.DenseInstance;

/**
 * This class extends the WEKA DenseInstance so it can be reused. This is an 
 * important feature to save time and memory when classifying iteratively many 
 * new instances.
 * 
 * @author Ignacio Arganda-Carreras (iargandacarreras@gmail.com)
 *
 */
public class ReusableDenseInstance extends DenseInstance{

	/** for serialization */
	static final long serialVersionUID = -2322621365960366756L;

	/**
	 * Construct reusable dense instance
	 * 
	 * @param weight instance weight
	 * @param attValues array of attribute values
	 */
	public ReusableDenseInstance(double weight, double[] attValues) {
		super(weight, attValues);
	}

	/**
	 * Set instance values
	 * 
	 * @param weight instance weight
	 * @param attValues array of attribute values
	 */
	public void setValues( double weight, double[] attValues )
	{
		super.m_Weight = weight;
		super.m_AttValues = attValues;
	}
	
	
}
