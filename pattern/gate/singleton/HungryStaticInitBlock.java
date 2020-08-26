public class Singleton{
    //1.构造器私有化
    private Singleton(){}
    private static Singleton instance;
    //2.静态初始化块
    static {
        instance=new Singleton();
    }
    //3.向外暴露静态公共方法
    public static Singleton getInstance(){
        return instance;
    }
}