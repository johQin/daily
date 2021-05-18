package builder.improve;

//指挥者,指定制作流程，返回最后的产品
public class HouseDirector {
    HouseBuilder houseBuilder = null;

    //houseBuilder的传入
    //一通过构造器，
    public HouseDirector(HouseBuilder houseBuilder){
        this.houseBuilder = houseBuilder;
    }
    //二可以通过setter
    public void setHouseBuilder(HouseBuilder houseBuilder){
        this.houseBuilder= houseBuilder;
    }
    //如何处理建造房子的流程，交给指挥者
    public House constructHouse(){
        //对于不同具体事物构建顺序可以不同，具体由指挥者决定
        houseBuilder.buildBase();
        houseBuilder.buildWall();
        houseBuilder.roofed();
        return houseBuilder.buildHouse();
    }
}
