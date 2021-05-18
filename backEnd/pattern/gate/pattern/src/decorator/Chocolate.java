package decorator;

public class Chocolate extends Decorator{
    public Chocolate(Drink drink){
        super(drink);
        setDes("Chocolate");
        setPrice(3.0f);
    }
}
