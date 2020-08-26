public class Singleton{
    private Singleton(){}
    private static Singleton instance;
    //用到的时候创建
    //加了线程同步排队synchronized
    //但效率低下
    public static synchronized Singleton getInstance(){
        if(instance==null){
            instance=new Singleton();
        }
        return instance;
    }
}