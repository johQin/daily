package observer.improve;


public class Client {
    public static void main(String[] args) {

        WeatherData weatherData = new WeatherData();
        CurrentConditons currentConditons=new CurrentConditons();
        weatherData.registerObserver(currentConditons);
        Baidu baidu = new Baidu();
        weatherData.registerObserver(baidu);

        //通知各个注册观察者，天气变化
        weatherData.setData(26,102,0.7f);
        //删除观察者
        weatherData.removeObserver(currentConditons);
        System.out.println("观察者数目变化后");
        weatherData.setData(21,106,0.3f);
    }
}
