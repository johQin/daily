package visitor;

public class Woman extends Person{
    @Override
    public void accept(Action action) {
        action.getWomanValuate(this);
    }
}
