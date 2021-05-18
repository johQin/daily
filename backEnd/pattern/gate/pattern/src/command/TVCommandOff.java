package command;

public class TVCommandOff implements Command{
    TVReceiver TV;
    public TVCommandOff(TVReceiver TV){
        this.TV = TV;
    }
    @Override
    public void execute() {
        TV.off();
    }
    @Override
    public void revoke(){
        TV.on();
    }
}
