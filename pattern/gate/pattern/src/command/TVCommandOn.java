package command;

public class TVCommandOn implements Command{
    TVReceiver TV;
    public TVCommandOn(TVReceiver TV){
        this.TV = TV;
    }
    @Override
    public void execute() {
        TV.on();
    }
    @Override
    public void revoke(){
        TV.off();
    }
}
