package prototype;

public class Client {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        //传统的方法
        Sheep sheep = new Sheep("tom", 1, "白色");

        Sheep sheep2 = new Sheep(sheep.getName(), sheep.getAge(), sheep.getColor());
        Sheep sheep3 = new Sheep(sheep.getName(), sheep.getAge(), sheep.getColor());
        //....

        System.out.println(sheep);
        System.out.println(sheep2);
        System.out.println(sheep3);
        //...
    }


}
