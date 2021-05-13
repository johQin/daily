package strategy.improve;

public class PekingDuck extends Duck {
    @Override
    public void displayInfo() {
        System.out.println("我是北京烤鸭");
    }

    public PekingDuck() {
        flyBehavior = new BadFlyBehavior();
    }
}
