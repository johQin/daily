package observer.improve;

public class CurrentConditons implements Observer{
    //温度，气压，湿度
    private float temperature;
    private float pressure;
    private float humidity;
    //更新气象数据，由weatherData调用，推送模式
    public void update(float temperature,float pressure,float humidity){
        this.temperature = temperature;
        this.pressure = pressure;
        this.humidity = humidity;
        display();
    }
    public void display(){
        System.out.println("current weather:temperature~"+temperature+"C,pressure~"+pressure+"kpa,humidity~"+humidity );
    }
}
