package Composite;

import java.util.ArrayList;
import java.util.List;

public class College extends OrganizationComponent{
    List<OrganizationComponent> organizationComponentList = new ArrayList<OrganizationComponent>();

    //输出college下面包含的学系
    @Override
    protected void print() {
        String name = getName();
        String departments = "";
        for (OrganizationComponent organizationComponent:organizationComponentList) {
            departments = departments + organizationComponent.getName() + ",";
        }
        System.out.println("学院名："+ name+","+"下属学系："+ departments);
    }

    @Override
    protected void add(OrganizationComponent organizationComponent) {
        organizationComponentList.add(organizationComponent);
    }

    @Override
    protected void remove(OrganizationComponent organizationComponent) {
        organizationComponentList.remove(organizationComponent);
    }

    public College(String name, String des) {
        super(name, des);
    }
    @Override
    public String getName(){
        return super.getName();
    }
    @Override
    public String getDes(){
        return super.getDes();
    }
}
