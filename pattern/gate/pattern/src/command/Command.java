package command;

public interface Command {
    //执行操作
    public void execute();
    //撤销操作
    public void revoke();
}
