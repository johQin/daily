package adapter.classAdapter;

//被适配的类
public class Voltage220V {
    //输出220v电压
    public Integer output220V(){
        Integer src = 220;
        System.out.println("输出电压src="+src+"V");
        return src;
    }
}
