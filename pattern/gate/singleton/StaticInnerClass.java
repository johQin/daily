public class Singleton{
    private Singleton(){}
    //1.在Singleton执行类装载的时候，内部类SingletonInstance的内部并不会执行类加载，从而实现懒加载
    private static class SingletonInstance{
        //这里给了一个final，类装载后，不再执行赋值操作，在一定程度上增强了单例
        private static final Singleton INSTANCE=new Singleton();
    }
    //2.当在使用getInstance的时候，它会去取静态内部类的静态属性，
    //这个时候就会导致静态内部类SingletonInstance进行装载，
    //JVM在装载类的时候是线程安全的，装载只执行一次，并实例化INSTANCE
    public static Singleton getInstance(){
        return SingletonInstance.INSTANCE;
    }
}