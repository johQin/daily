package facade;

public class Stereo {
    private static Stereo instance = new Stereo();
    public static Stereo getInstance(){
        return instance;
    }
    public void on(){
        System.out.println("Stereo on");
    }
    public void off(){
        System.out.println("Stereo off");
    }
    public void up(){
        System.out.println("Voice up");
    }
    public void down(){
        System.out.println("Voice down");
    }
}
