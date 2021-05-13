package mediator;

public class Client {
    public static void main(String[] args) {
        //创建一个中介者对象
        Mediator mediator = new ConcreteMediator();

        //创建Alarm 并且加入到  ConcreteMediator 对象的HashMap
        Alarm alarm = new Alarm(mediator, "alarm");
        TV tv = new TV(mediator,"TV");

        //让闹钟发出消息
        alarm.SendAlarm(0);
        alarm.SendAlarm(1);
    }
}
