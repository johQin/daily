package facade;

public class PopCorn {
    private static PopCorn instance = new PopCorn();
    public static PopCorn getInstance(){
        return instance;
    }
    public void on(){
        System.out.println("PopCorn on");
    }
    public void off(){
        System.out.println("PopCorn off");
    }
    public void pop(){
        System.out.println("PopCorn pop out");
    }
}
