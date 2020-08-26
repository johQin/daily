public class Singleton{
    //1.构造器私有化
    private Singleton(){}
    //2.类的内部创建对象
    private static final Singleton instance=new Singleton();
    //3.向外暴露静态公共方法
    public static Singleton getInstance(){
        return instance;
    }
}
public class HungryStaticVarible{
    public static void main(String[] args) {
        Singleton instance = Singleton.getInstance();
		Singleton instance2 = Singleton.getInstance();
		System.out.println(instance == instance2); // true
		System.out.println("instance.hashCode=" + instance.hashCode());
		System.out.println("instance2.hashCode=" + instance2.hashCode());
    }
}