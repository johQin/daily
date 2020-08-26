public class Singleton{
    private Singleton(){}
    //加volatile，线程共享变量值，有改变时立马，写入主存，在一定程度上有线程同步的效果
    private static volatile Singleton instance;
    //用到的时候创建
    public static Singleton getInstance(){
        if(instance==null){
            //加了线程同步排队synchronized
            synchronized(Singleton.class){
                if(instance==null){
                    instance=new Singleton();
                }
            }
        }
        //当第一次多线程A，B（或者更多）都通过了第一重检查，都进入排队阶段。
        //A先进入synchronized代码块，执行代码内容，使instance实例化。
        //B（或者更多）在排队等候后，同样再判断，但不执行实例化。
        //但是在之后线程访问中，在第一重检查就会拦下，
        //因为instance已经实例化，就不会再进入synchronized代码块，无需再执行排队等待操作。
        //效率提高
        return instance;
    }
}