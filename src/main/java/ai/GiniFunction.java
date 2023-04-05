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
package ai;

/**
 *
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Authors: Ignacio Arganda-Carreras (iarganda@mit.edu), 
 * 			Albert Cardona (acardona@ini.phys.ethz.ch)
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

/**
 * This class implements a split function based on the Gini coefficient
 *
 */
public class GiniFunction extends SplitFunction {

	/**
	 * Servial version ID
	 */
	private static final long serialVersionUID = 9707184791345L;
	/**
	 * number of random features to use
	 */
	private int numOfFeatures;
	/**
	 * random number generator
	 */
	private final Random random;
	/**
	 * helper class to hold attribute and class pair
	 */
	private AttributeClassPairList attributeClassPairList;

	public GiniFunction(int numOfFeatures, final Random random) {
		this.numOfFeatures = numOfFeatures;
		this.random = random;
		this.attributeClassPairList = new AttributeClassPairList();
	}

	public void init(Instances data, ArrayList<Integer> indices) {
		if (indices.isEmpty()) {
			this.allSame = true;
			return;
		}

		this.attributeClassPairList.init(data, indices, numOfFeatures, random);
		this.attributeClassPairList.calculateMinimumGini(data.numClasses(), indices.size());
		this.index = this.attributeClassPairList.getIndex();
		this.threshold = this.attributeClassPairList.getThreshold();
		this.allSame = this.attributeClassPairList.isAllSame();
	}

	@Override
	public boolean evaluate(Instance instance) {
		if (allSame)
			return true;
		else
			return instance.value(this.index) < this.threshold;
	}

	@Override
	public SplitFunction newInstance() {
		return new GiniFunction(this.numOfFeatures, this.random);
	}

	private static class AttributeClassPairList {

		private ArrayList<AttributeClassPair> list;
		private boolean allSame;
		private int index;
		private double threshold;

		public AttributeClassPairList() {
			this.list = new ArrayList<AttributeClassPair>();
			this.allSame = false;
			this.index = 0;
			this.threshold = 0;
		}

		public void init(Instances data, ArrayList<Integer> indices, int numOfFeatures, final Random random) {
			final int len = data.numAttributes();
			final int numElements = indices.size();
			final int numClasses = data.numClasses();
			final int classIndex = data.classIndex();

			ArrayList<Integer> allIndices = new ArrayList<Integer>();
			for (int i = 0; i < len; i++)
				if (i != classIndex)
					allIndices.add(i);

			for (int i = 0; i < numOfFeatures; i++) {
				// Select the random feature
				final int index = random.nextInt(allIndices.size());
				final int featureToUse = allIndices.get(index);
				allIndices.remove(index); // remove that element to prevent from repetitions

				list.clear();
				for (int j = 0; j < numElements; j++) {
					final Instance ins = data.get(indices.get(j));
					list.add(new AttributeClassPair(ins.value(featureToUse), (int) ins.value(classIndex)));
				}

				// Sort pairs in increasing order
				Collections.sort(list, new Comparator<AttributeClassPair>() {
					public int compare(AttributeClassPair o1, AttributeClassPair o2) {
						final double diff = o2.attributeValue - o1.attributeValue;
						if (diff < 0)
							return 1;
						else if (diff == 0)
							return 0;
						else
							return -1;
					}

					public boolean equals(Object o) {
						return false;
					}
				});
			}
		}

		public void calculateMinimumGini(int numClasses, int numElements) {
			double minimumGini = Double.MAX_VALUE;
			final int len = list.size();
			final double[] probLeft = new double[numClasses];
			final double[] probRight = new double[numClasses];
			final int[] numLeft = new int[numClasses];
			final int[] numRight = new int[numClasses];
			int[] sortedClassCounts = new int[numClasses];
			int totalLeft = 0;
			int totalRight = numElements;
			// count the class counts for the right side of the split
			for (int i = 0; i < len; i++) {
				final AttributeClassPair pair = list.get(i);
				final int cls = pair.classValue;
				sortedClassCounts[cls]++;
			}
			System.arraycopy(sortedClassCounts, 0, numRight, 0, numClasses);

			// iterate through the pairs and calculate gini index at each split
			for (int i = 0; i < len - 1; i++) {
				final AttributeClassPair pair = list.get(i);
				final double value = pair.attributeValue;
				final int cls = pair.classValue;
				numLeft[cls]++;
				numRight[cls]--;
				totalLeft++;
				totalRight--;

				// calculate gini index for the current split
				if (value != list.get(i + 1).attributeValue) {
					double leftGini = 1.0;
					double rightGini = 1.0;

					for (int j = 0; j < numClasses; j++) {
						final double p = (double) numLeft[j] / totalLeft;
						leftGini -= p * p;

						final double q = (double) numRight[j] / totalRight;
						rightGini -= q * q;

						probLeft[j] = p;
						probRight[j] = q;
					}

					final double giniIndex = leftGini * totalLeft + rightGini * totalRight;

					if (giniIndex < minimumGini) {
						minimumGini = giniIndex;
						index = i;
						threshold = (value + list.get(i + 1).attributeValue) / 2.0;
					}
				}
			}
			allSame = (minimumGini == Double.MAX_VALUE);
		}

		public int getIndex() {
			return index;
		}

		public double getThreshold() {
			return threshold;
		}

		public boolean isAllSame() {
			return allSame;
		}
	}

	private static class AttributeClassPair {
		public double attributeValue;
		public int classValue;

		public AttributeClassPair(double attributeValue, int classValue) {
			this.attributeValue = attributeValue;
			this.classValue = classValue;
		}
	}
}