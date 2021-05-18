package decorator;

public class Decorator extends Drink{
    private Drink decoratedObj;
    public Decorator(Drink decoratedObj){
        this.decoratedObj = decoratedObj;
    }

    @Override
    public Float cost() {
        return super.getPrice() + decoratedObj.cost();
    }

    @Override
    public String getDes() {
        return super.getDes()+""+super.getPrice()+"&&"+decoratedObj.getDes();
    }
}
