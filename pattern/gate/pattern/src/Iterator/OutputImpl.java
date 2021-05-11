package Iterator;

import java.util.Iterator;
import java.util.List;

public class OutputImpl {
    List<College> colleges;
    public OutputImpl(List<College> colleges){
        this.colleges = colleges;
    }
    //输出学院
    public void printCollege(){
        Iterator<College> iterator = colleges.iterator();
        while(iterator.hasNext()){
            College college = (College) iterator.next();
            System.out.println("====="+college.getName()+"====");
            printDepartment(college.createIterator());

        }
    }
    //输出学系
    public void printDepartment(Iterator iterator){
        while(iterator.hasNext()){
            Department department =(Department) iterator.next();
            System.out.println(department.getName());

        }

    }
}
