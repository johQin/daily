package decorator;

public class Milk extends Decorator{
    public Milk(Drink drink){
        super(drink);
        setDes("Milk");
        setPrice(4.0f);
    }
}
