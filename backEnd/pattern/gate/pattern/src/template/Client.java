package template;

public class Client {
    public static void main(String[] args) {
        SoyMilk redbeanSoyMilk= new RedbeanSoyMilk();
        redbeanSoyMilk.make();
        SoyMilk peanutSoyMilk = new PeanutSoyMilk();
        peanutSoyMilk.make();
    }
}
