package bridge;

public class FoldedPhone extends Phone{
    public FoldedPhone(Brand brand){
        super( brand );
    }
    //子类的call其实是调用的父类的call方法，而父类的call方法实际上是通过接口调用实现类的方法，
    //所以这里的父类其实是一个桥，搭起了两个实现类方法之间的桥
    public void call(){
        super.call();
        System.out.println("折叠式手机");
    }
    public void open(){
        super.open();
        System.out.println("折叠式手机");
    }
    public void close(){
        super.close();
        System.out.println("折叠式手机");
    }
}
