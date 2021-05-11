package command;

public class LightCommandOff implements Command{
    LightReceiver light;

    public LightCommandOff(LightReceiver light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.off();
    }

    @Override
    public void revoke() {
        light.on();
    }
}
