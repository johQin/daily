package strategy.tradition;

public class PekingDuck extends Duck{
    @Override
    public void displayInfo() {
        System.out.println("我是北京烤鸭");
    }

    @Override
    public void fly() {
        System.out.println("北京烤鸭不能飞");
    }
}
