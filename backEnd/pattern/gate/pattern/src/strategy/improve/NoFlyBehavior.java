package strategy.improve;

public class NoFlyBehavior implements FlyBehavior{
    @Override
    public void fly() {
        System.out.println("我不会飞行");
    }
}
