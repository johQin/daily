package bridge;

public class XiaoMi implements Brand{

    @Override
    public void call() {
        System.out.println("小米手机拨打电话");
    }
    @Override
    public void open() {
        System.out.println("小米手机开机");
    }
    @Override
    public void close() {
        System.out.println("小米手机关机");
    }
}
