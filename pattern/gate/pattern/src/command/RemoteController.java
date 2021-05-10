package command;

//每一个按钮就是一个命令
public class RemoteController {
    //开按钮的命令数组
    Command[] onCommands;
    Command[] offCommands;

    //执行撤销的命令
    Command[] revokeCommand;

    public RemoteController(){
        onCommands = new Command[5];
        offCommands = new Command[5];
        //初始化遥控器上所有按钮
        for(int i = 0; i < 5; i++){
            onCommands[i] = new NoCommand();
            offCommands[i] = new NoCommand();
        }
    }
}
