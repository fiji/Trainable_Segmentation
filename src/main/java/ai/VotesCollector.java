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

import java.util.concurrent.Callable;

import weka.core.Instances;
import weka.core.Utils;

/**
 * Used to retrieve the out-of-bag vote of an ensemble classifier for a single
 * instance. In classification, does not return the class distribution but only
 * class index of the dominant class.
 *
 * Implements callable so it can be run in multiple threads.
 *
 * Adapted by Ignacio Arganda-Carreras from the Fran Supek's version to work on BalancedRandomTree objects
 */
public class VotesCollector implements Callable<Double> 
{

	protected final BalancedRandomTree[] tree;
	protected final int instanceIdx;
	protected final Instances data;
	protected final boolean[][] inBag;


	public VotesCollector(
			BalancedRandomTree[] tree, 
			int instanceIdx,
			Instances data, 
			boolean[][] inBag) 
	{

		this.tree = tree;
		this.instanceIdx = instanceIdx;
		this.data = data;
		this.inBag = inBag;

	}


	/** Determine predictions for a single instance. */
	@Override
	public Double call() throws Exception 
	{

		double[] classProbs = null;
		double regrValue = 0;


		classProbs = new double[data.numClasses()];

		for (int treeIdx = 0; treeIdx < tree.length; treeIdx++) 
		{

			if (inBag[treeIdx][instanceIdx])
				continue;

			double[] curDist = tree[treeIdx].evaluate( data.instance(instanceIdx) );

			for ( int classIdx = 0; classIdx < curDist.length; classIdx++ )
				classProbs[ classIdx ] += curDist[ classIdx ];

		}

		double vote = Utils.maxIndex(classProbs);   // consensus - for classification

		return vote;

	}


}

