package facade;

public class HomeCinemaFacade {
    //定义各个子系统的对象
    private Light light;
    private Stereo stereo;
    private DVDPlayer dvdPlayer;
    private Projector projector;
    private Screen screen;
    private PopCorn popCorn;

    public HomeCinemaFacade() {
        this.light = Light.getInstance();
        this.stereo = Stereo.getInstance();
        this.dvdPlayer = DVDPlayer.getInstance();
        this.projector = Projector.getInstance();
        this.screen = Screen.getInstance();
        this.popCorn = PopCorn.getInstance();
    }

    //将家庭影院的整体运行分为四步，由这四步去调用各个子系统的接口
    public void ready(){
        popCorn.on();
        popCorn.pop();
        screen.down();
        projector.on();
        stereo.on();
        dvdPlayer.on();
        light.dim();
    }
    public void play(){
        dvdPlayer.play();
    }
    public void pause(){
        dvdPlayer.pause();
    }
    public void end(){
        popCorn.off();
        screen.up();
        projector.off();
        stereo.off();
        dvdPlayer.off();
        light.bright();
    }
}
