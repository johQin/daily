package adapter.classAdapter;

public class VoltageAdapter extends Voltage220V implements InterfaceVoltage5V{
    @Override
    public Integer output5V() {
        Integer src = output220V();
        Integer dist = src / 44 ;
        System.out.println("适配器已将src="+ src + "V转换为目标电压dist="+dist+"V");
        return dist;
    }
}
