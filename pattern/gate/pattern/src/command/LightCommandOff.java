package command;

public class LightCommandOff implements Command{
    LightReceiver light;

    public LightCommandOff(LightReceiver light) {
        this.light = light;
    }

    @Override
    public void excute() {
        light.off();
    }

    @Override
    public void revoke() {
        light.on();
    }
}
