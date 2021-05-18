package observer.improve;

import observer.tradition.CurrentConditions;

import java.util.ArrayList;

public class WeatherData implements Subject{
    private float temperature;
    private float pressure;
    private float humidity;
    private ArrayList<Observer> observers;

    public WeatherData() {
        this.observers = new ArrayList<Observer>();
    }
    //注册一个观察者
    @Override
    public void registerObserver(Observer o) {
        observers.add(o);
    }
    //移除一个观察者
    @Override
    public void removeObserver(Observer o) {
        if(observers.contains(o)){
            observers.remove(o);
        }
    }

    @Override
    public void notifyObservers() {
        for(int i = 0 ;i<observers.size();i++){
            observers.get(i).update(this.temperature,this.pressure,this.humidity);
        }
    }
    //将最新的信息推送给客户端/接入方
    public void dataChange(){
        notifyObservers();
    }
    //当数据有更新时，就调用setData，
    public void setData(float temperature,float pressure,float humidity){
        this.temperature = temperature;
        this.pressure = pressure;
        this.humidity = humidity;
        dataChange();
    }


}
