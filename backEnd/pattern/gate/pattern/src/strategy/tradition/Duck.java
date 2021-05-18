package strategy.tradition;

public abstract class Duck {
    public void yell(){
        System.out.println("鸭子嘎嘎叫");
    }
    public void swim(){
        System.out.println("鸭子会游泳");
    }
    public void fly(){
        System.out.println("鸭子会飞翔");
    }
    public abstract void displayInfo();
}
