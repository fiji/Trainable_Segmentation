package trainableSegmentation_clij.utils;

public abstract class Neighborhood2D {
	
	protected trainableSegmentation_clij.utils.Cursor2D cursor;
	
	public abstract Iterable<trainableSegmentation_clij.utils.Cursor2D> getNeighbors(  );
	
	public void setCursor( trainableSegmentation_clij.utils.Cursor2D cursor )
	{
		this.cursor = cursor;
	}
	
	public Cursor2D getCursor()
	{
		return this.cursor;
	}
	
}
