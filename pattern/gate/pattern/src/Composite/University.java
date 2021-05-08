package Composite;

import java.util.ArrayList;
import java.util.List;

//University就是Composite，可以管理College
public class University extends OrganizationComponent{

    List<OrganizationComponent> organizationComponentList = new ArrayList<OrganizationComponent>();

    //输出University下面包含的学院
    @Override
    protected void print() {
        String name = getName();
        String colleges = "";
        for (OrganizationComponent organizationComponent:organizationComponentList) {
            colleges = colleges + organizationComponent.getName() + ",";
        }
        System.out.println("大学名："+ name+","+"下属学院："+ colleges);
    }

    @Override
    protected void add(OrganizationComponent organizationComponent) {
        organizationComponentList.add(organizationComponent);
    }

    @Override
    protected void remove(OrganizationComponent organizationComponent) {
        organizationComponentList.remove(organizationComponent);
    }

    public University(String name, String des) {
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
