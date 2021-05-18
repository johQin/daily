package mediator;

public class TV extends Colleague{
    public TV(Mediator mediator,String name){
        super(mediator,name);
        //在创建Alarm同事对象
        mediator.Register( name,this);
    }
    public void SendAlarm(int stateChange){
        SendMessage(stateChange);
    }

    @Override
    public void SendMessage(int stateChange) {
        //调用中介者对象的getMessage
        this.GetMediator().GetMessage(stateChange,this.name);
    }
    public void startTV(){
        System.out.println("TV starting");
    }
}
