package builder.improve;

public class CommonHouse extends HouseBuilder{

    @Override
    public void buildBase() {
        System.out.println("普通房子地基深5m");
    }

    @Override
    public void buildWall() {
        System.out.println("普通房子墙厚20cm");
    }

    @Override
    public void roofed() {
        System.out.println("普通房子封平顶");
    }
}
