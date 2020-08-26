public class Inversion{
    public static void main(String[] args) {
        Person person=new Person();
        person.receive(new Email());
        person.receive(new Wechat());
    }
}

/**
 * 完成Person接收消息的功能
 * 方式1：
 * 1. 传给receive一个Eamil对象，通过调用Email的getInfo方法，返回邮件消息。简单，比较容易想到,
 * 2. 如果我们获取的对象是微信，短信等等，则新增类，同时Person也要增加相应的接收方法
 * 3. 解决思路：引入一个抽象的接口IRceiver，表示接受者，这样Person类与接口IReceiver发生依赖
 * 4. 因为Email，weixin等等属于接收的范围，他们各自实现IReceiver 接口就ok，这样我们就符合依赖倒转原则
 */
class Person{
    public void receive(IReceiver iReceiver){
        System.out.println(iReceiver.getInfo());
    }
}
interface IReceiver{
    public String getInfo();
}
class Email implements IReceiver{
    public String getInfo(){
        return "邮件消息：hello Email";
    }
}
class Wechat implements IReceiver{
    public String getInfo(){
        return "微信消息：hello Wechat";
    }
}