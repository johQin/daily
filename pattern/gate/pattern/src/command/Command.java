package command;

public interface Command {
    //执行操作
    public void excute();
    //撤销操作
    public void revoke();
}
