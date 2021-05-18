package Iterator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class InfoCollege implements College{
    List<Department> departmentList;

    public InfoCollege() {
        this.departmentList = new ArrayList<Department>();
        addDepartment("information security","信息安全");
        addDepartment("network engineer","网络工程");
        addDepartment("information counter","信息对抗");
    }

    @Override
    public String getName() {
        return "信息工程学院";
    }

    @Override
    public void addDepartment(String name, String desc) {
        Department department = new Department(name, desc);
        departmentList.add(department);
    }

    @Override
    public Iterator createIterator() {
        return new InfoCollegeIterator(departmentList);
    }
}
