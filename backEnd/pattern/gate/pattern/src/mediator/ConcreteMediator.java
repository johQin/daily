package mediator;

import java.util.HashMap;

public class ConcreteMediator extends Mediator{
    //集合，放入所有的同事对象
    private HashMap<String, Colleague> colleagueMap;
    private HashMap<String, String> interMap;

    public ConcreteMediator() {
        colleagueMap = new HashMap<String, Colleague>();
        interMap = new HashMap<String, String>();
    }
    @Override
    public void Register(String colleagueName, Colleague colleague) {

        colleagueMap.put(colleagueName, colleague);
        if (colleague instanceof Alarm) {
            interMap.put("Alarm", colleagueName);
        } else if (colleague instanceof TV) {
            interMap.put("TV", colleagueName);
        }
    }
    public void GetMessage(int stateChange,String colleagueName){
        if (colleagueMap.get(colleagueName) instanceof Alarm){
            if(stateChange ==0){
                ((TV) (colleagueMap.get(interMap.get("TV")))).startTV();
            }else{
                //其他设备
                System.out.println("协调其他设备");
            }
        }
    }
    @Override
    public void SendMessage() {

    }
}
