package command;

//每一个按钮就是一个命令
public class RemoteController {
    //开按钮的命令数组
    Command[] onCommands;
    Command[] offCommands;

    //执行撤销的命令
    Command revokeCommand;
    //初始化所有按钮的命令为空命令
    public RemoteController(){
        onCommands = new Command[5];
        offCommands = new Command[5];
        //初始化遥控器上所有按钮
        for(int i = 0; i < 5; i++){
            onCommands[i] = new NoCommand();
            offCommands[i] = new NoCommand();
        }
    }
    //给按钮设置我们需要的命令
    public void setCommand(int number, Command commandOn,Command commandOff){
        onCommands[number] = commandOn;
        offCommands[number] = commandOff;
    }
    //按下on按钮
    public void onButtonWasPushed(int number){
        //找到你按下的按钮，并调用对应的方法
        onCommands[number].execute();
        //记录这次操作用于撤销
        revokeCommand = onCommands[number];
    }
    //按下off按钮
    public void offButtonWasPushed(int number){
        offCommands[number].execute();
        revokeCommand = offCommands[number];
    }
    //按下撤销按钮
    public void revokeButtonWasPushed(){
        revokeCommand.revoke();
    }

}
