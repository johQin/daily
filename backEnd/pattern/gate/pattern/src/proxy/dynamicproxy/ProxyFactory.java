package proxy.dynamicproxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

public class ProxyFactory {
    //维护一个目标对象，被代理对象
    private Object target;
    //构造器，初始化目标对象
    public ProxyFactory(Object target) {
        this.target = target;
    }
    //给目标对象生成代理对象
    public Object getProxyInstance(){
        //static Object newProxyInstance(ClassLoader loader,Class<?>[] interfaces, InvocationHandler h)
        //1. ClassLoader：指定当前目标对象使用的类加载器，获取加载器的方法固定
        //2. Class<?>[]：目标对象实现的接口类型，使用泛型方法确认类型
        //3. InvocationHandler：事情处理，确定执行目标对象的什么方法。
        return Proxy.newProxyInstance(
                target.getClass().getClassLoader(),
                target.getClass().getInterfaces(),
                new InvocationHandler() {
                    @Override
                    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                        System.out.println("代理开始");
                        //利用反射机制调用目标对象方法
                        Object backVal = method.invoke(target,args);
                        System.out.println("代理结束");
                        return backVal;
                    }
                }
        );
    }
}
