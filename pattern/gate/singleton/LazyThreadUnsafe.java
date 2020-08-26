public class Singleton{
    private Singleton(){}
    private static Singleton instance;
    //用到的时候创建
    public static Singleton getInstance(){
        //问题：如果在多线程下，
        //一个线程进入了if (singleton == null)判断语句块，还未来得及往下执行，
        //另一个线程也通过了这个判断语句，这时便会产生多个实例。
        if(instance==null){
            instance=new Singleton();
        }
        return instance;
    }
}