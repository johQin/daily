package observer.tradition;

public class Client {
    public static void main(String[] args) {
        //创建客户端
        CurrentConditions currentConditions = new CurrentConditions();
        //创建WeatherData对象，并聚合客户端对象
        WeatherData weatherData = new WeatherData(currentConditions);
        //更新天气情况，并推送
        weatherData.setData(21,101, 0.5f);

        //天气变化
        weatherData.setData(15,110, 0.4f);
    }

}
