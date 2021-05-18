package facade;

public class DVDPlayer {
    //单例模式，饿汉式
    private static DVDPlayer instance = new DVDPlayer();
    public static DVDPlayer getInstance(){
        return instance;
    }
    public void on(){
        System.out.println("DVD on");
    }
    public void off(){
        System.out.println("DVD off");
    }
    public void play(){
        System.out.println("DVD playing");
    }
    public void pause(){
        System.out.println("DVD pause");
    }
    //....等等其他方法
}
