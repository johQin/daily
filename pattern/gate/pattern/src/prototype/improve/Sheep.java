package prototype.improve;

public class Sheep implements Cloneable{
    private String name;
    private int age;
    private String color;
    //这里用于验证直接利用clone方法，仅仅实现的是浅拷贝
    private Sheep friend;

    //动态增加的属性，不会像传统创建模式一样，需要在new处增加get
    private String from = "SiChuan";
    //记得在toString那里更新

    public Sheep(String name, int age, String color){
        super();
        this.name = name;
        this.age = age;
        this.color = color;
    }
    //克隆该实例，使用默认的clone方法来完成
    //浅拷贝
    @Override
    protected Object clone() {
        Sheep sheep= null;
        try{
            //super.clone()可能会抛出异常，所以要么在方法上抛错throw，要么在当前位置进行try catch
            sheep = (Sheep) super.clone();
        }catch (Exception e){
            System.out.println(e.getMessage());
        }

        return sheep;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public Sheep getFriend() {
        return friend;
    }

    public void setFriend(Sheep friend) {
        this.friend = friend;
    }

    @Override
    public String toString() {
        return "Sheep{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", color='" + color + '\'' +
                ", from='" + from + '\'' +
                '}';
    }
}
