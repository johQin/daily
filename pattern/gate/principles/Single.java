public class Single{
    public static void main(String[] args){
        //方案1，违反单一职责原则
        Vehicle vehicle=new Vehicle();
        vehicle.run("汽车");
        vehicle.run("轮船");
        vehicle.run("飞机");

        //方案2，符合单一职责原则，类级别开销大
        AirVehicle ah=new AirVehicle();
        ah.run("飞机");
        WaterVehicle wh=new WaterVehicle();
        wh.run("轮船");

        //方案3，符合单一职责，采用三个不同的方法，开销小
        Vehicle3Method v3=new Vehicle3Method();
        v3.runAir("飞机");
        v3.runWater("轮船");
        v3.runRoad(4,);
    }
}
//交通工具类

/**
 * 方案1
 * run方法中违反了单一职责原则，根据交通工具运行方法不同，分解成不同的类或不同的方法
 */
class Vehicle{
    public void run(String vehicle){
        System.out.println(vehicle+"在地上跑");
    }
}

/**
 * 方案2
 * 遵守了单一职责原则，但这样的改动很大，
 */
class RoadVehicle{
    public void run(String vehicle){
        System.out.println(vehicle+"在地上运行");
    }
}
class WaterVehicle{
    public void run(String vehicle){
        System.out.println(vehicle+"在水中运行");
    }
}
class AirVehicle{
    public void run(String vehicle){
        System.out.println(vehicle+"在天上运行");
    }
}

class Vehicle3Method{
    public void runWater(String vehicle){
        System.out.println(vehicle+"在水中运行");
    }
    public void runRoad(Integer time,String vehicle){
        if(time>5 && time<12){
            System.out.println(vehicle+"在地上运行");
        }else{
            System.out.println("抱歉收费站已关闭")
        }
        
    }
    public void runAir(String vehicle){
        System.out.println(vehicle+"在天上运行");
    }
}