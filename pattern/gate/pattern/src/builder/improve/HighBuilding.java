package builder.improve;

public class HighBuilding extends HouseBuilder{

    @Override
    public void buildBase() {
        System.out.println("普通房子地基深10m");
    }

    @Override
    public void buildWall() {
        System.out.println("普通房子墙厚40cm");
    }

    @Override
    public void roofed() {
        System.out.println("普通房子封圆顶");
    }
}
