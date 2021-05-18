package strategy.improve;

public class WildDuck extends Duck {
    public WildDuck() {
        flyBehavior = new GoodFlyBehavior();
    }

    @Override
    public void displayInfo() {
        System.out.println("我是野鸭");
    }
    public void setFlyBehavior(FlyBehavior flyBehavior){
        this.flyBehavior = flyBehavior;
    }
}
