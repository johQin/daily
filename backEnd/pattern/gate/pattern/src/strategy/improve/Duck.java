package strategy.improve;

public abstract class Duck {
    //属性，策略接口
    FlyBehavior flyBehavior;
    public void yell(){
        System.out.println("鸭子嘎嘎叫");
    }
    public void swim(){
        System.out.println("鸭子会游泳");
    }
    public void fly(){
        if(flyBehavior!=null){
            flyBehavior.fly();
        }
    }
    public abstract void displayInfo();
}
