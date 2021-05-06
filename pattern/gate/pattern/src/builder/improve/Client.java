package builder.improve;

public class Client {

    public static void main(String[] args) {
        //盖普通房子
        CommonHouse commonHouse = new CommonHouse();
        //准备指挥者
        HouseDirector houseDirector = new HouseDirector(commonHouse);
        //完成盖房，返回产品
        House house = houseDirector.constructHouse();
        System.out.println("-----------");
        //盖高楼
        HighBuilding highBuilding = new HighBuilding();
        houseDirector.setHouseBuilder(highBuilding);
        House house1 = houseDirector.constructHouse();
    }



}
