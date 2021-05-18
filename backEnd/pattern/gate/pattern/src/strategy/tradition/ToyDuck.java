package strategy.tradition;

public class ToyDuck extends Duck{
    @Override
    public void yell() {
        System.out.println("我没有此功能");
    }

    @Override
    public void swim() {
        System.out.println("我没有动能去游泳");
    }

    @Override
    public void fly() {
        System.out.println("我只能浮在水面上");
    }

    @Override
    public void displayInfo() {
        System.out.println("我是玩具鸭");
    }
}
