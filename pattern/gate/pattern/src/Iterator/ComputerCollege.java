package Iterator;

import java.util.Iterator;

public class ComputerCollege implements College{
    Department[] departments;
    int numOfDepartment = 0;

    public ComputerCollege() {
        this.departments = new Department[5];
        addDepartment("softwart engineer","软件工程");
        addDepartment( "algorithm","算法");
        addDepartment( "artificial intelligence","人工智能");
//        addDepartment( "big data","大数据");
//        addDepartment( "cloud compute","云计算");

    }

    @Override
    public String getName() {
        return "计算机学院";
    }

    @Override
    public void addDepartment(String name, String desc) {
        Department department = new Department(name,desc);
        departments[numOfDepartment] = department;
        numOfDepartment += 1;

    }

    @Override
    public Iterator createIterator() {
        return new ComputerCollegeIterator(departments);

    }
}
