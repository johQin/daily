package template.hook;

public class Client {
    public static void main(String[] args) {
        System.out.println("======制作红豆豆浆=======");
        SoyMilk redbeanSoyMilk= new RedbeanSoyMilk();
        redbeanSoyMilk.make();
        System.out.println("======制作花生豆浆=======");
        SoyMilk peanutSoyMilk = new PeanutSoyMilk();
        peanutSoyMilk.make();
        System.out.println("======制作原味豆浆=======");
        SoyMilk pureSoyMilk = new PureSoyMilk();
        pureSoyMilk.make();
    }
}
