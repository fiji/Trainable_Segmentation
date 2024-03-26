/*-
 * #%L
 * Fiji distribution of ImageJ for the life sciences.
 * %%
 * Copyright (C) 2010 - 2024 Fiji developers.
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
package trainableSegmentation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import org.junit.Test;

import ij.ImagePlus;
import ij.gui.Roi;

public class ExamplesTest {
	@Test
	public void testExamples() {
		final URL url = getClass().getResource("/bridge.png");
		final ImagePlus bridge = new ImagePlus(url.getPath());;
		assertNotNull(bridge);

		final ArrayList<Roi> expected_examples = new ArrayList<Roi>();
		final Roi roi = new Roi(100, 100, 30, 30);
		expected_examples.add(roi);

		WekaSegmentation segmentator = new WekaSegmentation(bridge);
		segmentator.addClass();
		assertEquals(3, segmentator.getNumOfClasses());
		segmentator.setClassLabel(2, "label");
		segmentator.addExample(0, new Roi(10, 10, 50, 50), 1);
		segmentator.addExample(1, new Roi(400, 400, 30, 30), 1);
		segmentator.addExample(2, roi, 1);
		try {
			final Path path = Files.createTempFile("tws-", ".twse.gz");
			segmentator.saveExamples(path.toString());

			segmentator.removeClass(0);
			assertEquals(2, segmentator.getNumOfClasses());
			assertEquals("class 2", segmentator.getClassLabel(0));
			assertEquals("label", segmentator.getClassLabel(1));
			assertEquals("class 3", segmentator.getClassLabel(2));
			assertEquals(expected_examples, segmentator.getExamples(1, 1));

			segmentator = new WekaSegmentation(bridge);
			assertEquals(2, segmentator.getNumOfClasses());
			segmentator.loadExamples(path.toString());
			assertEquals(3, segmentator.getNumOfClasses());
			assertEquals("label", segmentator.getClassLabel(2));
			assertEquals(expected_examples, segmentator.getExamples(2, 1));

			Files.delete(path);
		} catch (IOException e) {
			e.printStackTrace();
			fail("Caught IOException");
		}
	}
}
