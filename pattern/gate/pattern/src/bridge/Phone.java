package bridge;

public abstract class Phone {

    //组合品牌
    private Brand brand;
    //构造器实现组合
    public Phone(Brand brand){
        super();
        this.brand = brand;
    }
    protected void call(){
        brand.call();
    }
    protected void open(){
        brand.open();
    }
    protected void close(){
        brand.close();
    }
}
