package adapter.objAdapter;

public class Phone {
    public void charge(InterfaceVoltage5V interfaceVoltage5V){
        Integer workVoltage = interfaceVoltage5V.output5V();
        if (workVoltage == 5){
            System.out.println("充电工作电压正常，正充电");
        }else{
            System.out.println("充电工作电压异常，请检查适配器");    
        }
    }
}
