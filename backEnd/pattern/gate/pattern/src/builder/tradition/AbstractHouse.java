package builder.tradition;

public abstract class AbstractHouse {
    public abstract void buildBasic();
    public abstract void buildWall();
    public abstract void roofed();
    public void build(){
        buildBasic();
        buildWall();
        roofed();
    }
}
