package proxy.dynamicproxy;

public class Client {
    public static void main(String[] args) {
        //创建目标对象
        ITeacherDao target = new TeacherDao();
        //给目标对象，创建代理对象，可以转为ITeacherDao
        ITeacherDao proxyInstance  = (ITeacherDao) new ProxyFactory(target).getProxyInstance();
        System.out.println("proxyInstance= "+ proxyInstance.getClass());//class com.sun.proxy.$Proxy0
        //通过代理对象，调用目标对象的方法
        proxyInstance.teach();
        proxyInstance.checkHomework(10);


    }
}
