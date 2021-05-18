package strategy.improve;

public class BadFlyBehavior implements FlyBehavior{
    @Override
    public void fly() {
        System.out.println("我的飞行本领很菜");
    }
}
