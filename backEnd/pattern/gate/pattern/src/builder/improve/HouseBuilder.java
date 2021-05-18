package builder.improve;

public abstract class HouseBuilder {
    protected House house=new House();

    //将建造的流程写好，抽象方法
    public abstract void buildBase();
    public abstract void buildWall();
    public abstract void roofed();

    //房子建好后，将产品（房子）返回
    public House buildHouse(){
        return house;
    }
}
