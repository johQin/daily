package strategy.improve;

public class ToyDuck extends Duck {
    public ToyDuck() {
        flyBehavior = new NoFlyBehavior();
    }
    @Override
    public void yell() {
        System.out.println("我没有此功能");
    }

    @Override
    public void swim() {
        System.out.println("我没有动能去游泳");
    }


    @Override
    public void displayInfo() {
        System.out.println("我是玩具鸭");
    }
    public void setFlyBehavior(FlyBehavior flyBehavior){
        this.flyBehavior = flyBehavior;
    }
}
