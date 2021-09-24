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
package trainableSegmentation.metrics;

public class InformationStatistics 
{
	/** entropy of ground truth T */
	public double ht = 0;
	/** entropy of predicted segmentation S */
	public double hs = 0;
	
	/** conditional entropy of T given S */
	public double hts = 0;
	/** conditional entropy of S given T */
	public double hst = 0;
		
	/** precision: C(T|S) = ( H(T) - H(T|S) ) / H(T) */
	public double precision = 0;
	
	/** recall C(S|T) = ( H(S) - H(S|T) ) / H(S) */
	public double recall = 0;
	
	/** F-score, harmonic mean of precision and recall */
	public double fScore = 0;
	
	/** value of the variation of information */
	public double vi = 0;
	
	/**
	 * Create information statistics
	 * 
	 * @param ht entropy of ground truth T
	 * @param hs entropy of predicted segmentation S	 
	 * @param hts conditional entropy of T given S
	 * @param hst conditional entropy of S given T	 
	 * @param vi value of the variation of information
	 */
	public InformationStatistics(
			double ht,
			double hs,
			double hts,
			double hst,			
			double vi)
	{
		this.ht = ht;
		this.hs = hs;
		this.hst = hst;
		this.hts = hts;
		this.vi = vi;
		
		// An information theoretic analog of Rand precision 
		// is the asymmetrically normalized mutual information
		// C(T|S) (T = ground truth, S = segmentation)
		this.precision = (ht - hts) / ht;

		// An information theoretic analog of Rand recall 
		// is defined similarly
		// C(S|T) (T = ground truth, S = segmentation)
		this.recall = (hs - hst) / hs;

		if( this.recall == 0 )
			this.precision = 1.0;
		
		if( (precision + recall) > 0)
			this.fScore = 2.0 * precision * recall / ( precision + recall );				
	}

}
