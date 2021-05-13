package strategy.improve;

public class Client {
    public static void main(String[] args) {
        WildDuck wildDuck = new WildDuck();
        wildDuck.fly();
        PekingDuck pekingDuck = new PekingDuck();
        pekingDuck.fly();
        ToyDuck toyDuck = new ToyDuck();
        toyDuck.fly();
    }
}
