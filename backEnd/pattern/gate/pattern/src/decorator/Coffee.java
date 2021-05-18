package decorator;

public class Coffee extends Drink{
    @Override
    public Float cost() {
        return super.getPrice();
    }
}
