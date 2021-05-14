package strategy.improve;

public class Client {
    public static void main(String[] args) {
        WildDuck wildDuck = new WildDuck();
        wildDuck.fly();
        PekingDuck pekingDuck = new PekingDuck();
        pekingDuck.fly();
        ToyDuck toyDuck = new ToyDuck();
        toyDuck.fly();
        //动态改变对象的行为
        toyDuck.setFlyBehavior(new BadFlyBehavior());
        toyDuck.fly();
    }
}
