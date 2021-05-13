package mediator;

public abstract class Mediator {
    //将中介者对象，加入到集合中
    public abstract void Register(String ColleagueName,  Colleague colleague);
    //接受消息，具体的同事对象发出
    public abstract void GetMessage(int stateChange, String colleague);
    public abstract void SendMessage();
}
