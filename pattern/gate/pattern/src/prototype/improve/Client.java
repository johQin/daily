package prototype.improve;

public class Client {
    public static void main(String[] args){
        //原型模式完成对象的创建
        //使用默认的克隆方法
        Sheep sheep = new Sheep("zy",18,"pureWhite");
        Sheep friend = new Sheep("qkh",24, "skyblue");
        sheep.setFriend(friend);
        Sheep sheep1 = (Sheep) sheep.clone();
        Sheep sheep2 = (Sheep) sheep.clone();
        //验证浅拷贝
        System.out.println(sheep1+", sheep1.friend="+sheep1.getFriend().hashCode());
        System.out.println(sheep2+", sheep2.friend="+sheep2.getFriend().hashCode());
        //发现两个拷贝friend的hashcode值都是一样的，所以是浅拷贝。
    }
}
