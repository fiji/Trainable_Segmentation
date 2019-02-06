package trainableSegmentation;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.Test;

import ij.ImagePlus;
import ij.process.FloatProcessor;

public class TestGaborThreads {
	
	static private final boolean test1() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);
		FeatureStack fs = new FeatureStack(original);
		
		ExecutorService exec = Executors.newFixedThreadPool(8);
		try {
			fs.getGabor(original, 4, 1.0, 1.0, 1.0, 4, exec);
		} finally {
			exec.shutdown();
		}
		
		return printThreads();
	}
	
	static private final boolean test2() {
		float[] pixels = new float[100 * 100];
		for (int i=0; i<pixels.length; ++i) pixels[i] = (float)Math.random() * 255;
		FloatProcessor fp = new FloatProcessor(100, 100, pixels);
		ImagePlus original = new ImagePlus("original", fp);
		FeatureStack fs = new FeatureStack(original);

		fs.setEnabledFeature("Gabor", true);
		fs.updateFeaturesMT(8);
		fs.shutDownNow();
		
		printThreads();
		try {
			// Give it time to remove the threads
			Thread.sleep(500);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return printThreads();
	}

	private static boolean printThreads() {
		ThreadGroup rootGroup = Thread.currentThread().getThreadGroup();
		ThreadGroup parentGroup;
		while ((parentGroup = rootGroup.getParent()) != null) {
		    rootGroup = parentGroup;
		}
		Thread[] threads = new Thread[rootGroup.activeCount()];
		while (rootGroup.enumerate(threads, true ) == threads.length) {
		    threads = new Thread[threads.length * 2];
		}
		int count = 0;
		int alive = 0;
		for (int i=0; i<threads.length; ++i) {
			if (null == threads[i]) break;
			++count;
			if (threads[i].isAlive()) ++alive;
		}
		System.out.println("Number of Thread instances found: " + count + " of which alive: " + alive);
		return count == alive;
	}
	
	@Test
	public void test() {
		assert test1();
		assert test2();
	}
}
