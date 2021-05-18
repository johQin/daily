package observer.tradition;

//1. 包含最新天气情况
//2. 包含CurrentConditions 对象
//3. 当数据由更新时，就主动的调用，CurrentConditions对象update方法
public class WeatherData {
    private float temperature;
    private float pressure;
    private float humidity;
    private CurrentConditions currentConditions;

    public WeatherData(CurrentConditions currentConditions) {
        this.currentConditions = currentConditions;
    }

    //将最新的信息推送给客户端/接入方
    public void dataChange(){
        currentConditions.update(temperature,pressure,humidity);
    }
    //当数据有更新时，就调用setData，
    public void setData(float temperature,float pressure,float humidity){
        this.temperature = temperature;
        this.pressure = pressure;
        this.humidity = humidity;
        dataChange();
    }
}
