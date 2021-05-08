package Composite;

public abstract class OrganizationComponent {
    private String name;
    private String des;

    //因为叶子节点是无须实现这个方法的，所以这里写一个空实现而不是abstract方法
    protected void add(OrganizationComponent organizationComponent){
        //默认实现
        throw new UnsupportedOperationException();
    }
    protected void remove(OrganizationComponent organizationComponent){
        //默认实现
        throw new UnsupportedOperationException();
    }
    //下面所有的子类都需要实现此方法
    protected abstract void print();

    public OrganizationComponent(String name,String des){
        this.name = name;
        this.des = des;
    }

    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public String getDes() {
        return des;
    }
    public void setDes(String des) {
        this.des = des;
    }
}
