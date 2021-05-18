package facade;

public class Light {
    private static Light instance = new Light();
    public static Light getInstance(){
        return instance;
    }
    public void on(){
        System.out.println("Light on");
    }
    public void off(){
        System.out.println("Light off");
    }
    public void dim(){
        System.out.println("Light dim");
    }
    public void bright(){
        System.out.println("Light bright");
    }

}
