package ai;

/**
 * Inner class to order attributes while preserving class indices 
 *
 */
public class AttributeClassPair
{
	/** real value of the corresponding attribute */
	protected double attributeValue;
	/** index of the class associated to this pair */
	protected int classValue;
	/**
	 * Create pair attribute-class
	 * 
	 * @param attributeValue real attribute value
	 * @param classIndex index of the class associated to this sample
	 */
	AttributeClassPair(double attributeValue, int classIndex)
	{
		this.attributeValue = attributeValue;
		this.classValue = classIndex;
	}
}
