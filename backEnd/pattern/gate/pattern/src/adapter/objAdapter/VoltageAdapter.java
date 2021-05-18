package adapter.objAdapter;

public class VoltageAdapter implements InterfaceVoltage5V {
    private Voltage220V voltage220V;
    @Override
    public Integer output5V() {
        Integer dist = 0;
        if(null!=voltage220V){
            Integer src = voltage220V.output220V();
            dist = src / 44 ;
            System.out.println("对象适配器已将src="+ src + "V转换为目标电压dist="+dist+"V");
        }
        return dist;
    }
    public VoltageAdapter(Voltage220V voltage220V){
        this.voltage220V = voltage220V;
    }
}
