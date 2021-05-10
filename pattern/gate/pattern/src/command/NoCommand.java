package command;

//没有任何命令，即空执行，用于初始化所有按钮，当调用空命令时，对象什么也不做。
//其实，这也是一种设计模式，可以省掉对空的命令的判断
public class NoCommand implements Command{
    @Override
    public void excute() {

    }

    @Override
    public void revoke() {

    }
}
