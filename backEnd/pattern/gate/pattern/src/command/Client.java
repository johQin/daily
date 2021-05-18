package command;

public class Client {
    public static void main(String[] args) {

        //使用命令模式，完成通过遥控器，对点灯的操作
        //创建电灯命令的接收者
        LightReceiver light = new LightReceiver();

        //创建电灯相关命令
        Command lightOn = new LightCommandOn(light);
        Command lightOff = new LightCommandOff(light);

        //创建命令调用者
        RemoteController remoteController = new RemoteController();
        //设置遥控器上按钮对应的命令，例如number:0就是点灯的开和关的操作
        remoteController.setCommand(0,lightOn,lightOff);
        System.out.println("按下电灯开按钮");
        remoteController.onButtonWasPushed(0);
        System.out.println("按下电灯关按钮");
        remoteController.offButtonWasPushed(0);
        System.out.println("按下命令撤销按钮");
        remoteController.revokeButtonWasPushed();

        TVReceiver TV = new TVReceiver();

        Command TVOn = new TVCommandOn(TV);
        Command TVOff = new TVCommandOff(TV);

        remoteController.setCommand(1,TVOn,TVOff);

        System.out.println("按下电视机开按钮");
        remoteController.onButtonWasPushed(1);
        System.out.println("按下电视机关按钮");
        remoteController.offButtonWasPushed(1);
        System.out.println("按下命令撤销按钮");
        remoteController.revokeButtonWasPushed();



    }
}
