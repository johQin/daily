package observer.improve;

public class Baidu implements Observer{

    private float temperature;
    private float pressure;
    private float humidity;

    public void update(float temperature,float pressure,float humidity){
        this.temperature = temperature;
        this.pressure = pressure;
        this.humidity = humidity;
        display();
    }
    public void display(){
        System.out.println("百度 current weather:temperature~"+temperature+"C,pressure~"+pressure+"kpa,humidity~"+humidity );
    }
}
