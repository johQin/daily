package decorator;

public abstract class Drink {
    public String des;//描述
    private Float price = 0.0f;

    public String getDes() {
        return des;
    }
    public void setDes(String des) {
        this.des = des;
    }
    public Float getPrice() {
        return price;
    }
    public void setPrice(Float price) {
        this.price = price;
    }
    //计算费用的抽象方法
    //由子类实现
    public abstract Float cost();
}
