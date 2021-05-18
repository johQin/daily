package command;

public class LightCommandOn implements Command{

    LightReceiver light;

    public LightCommandOn(LightReceiver light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.on();
    }

    @Override
    public void revoke() {
        light.off();
    }
}
