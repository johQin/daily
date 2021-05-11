package Iterator;

import java.util.Iterator;

public class ComputerCollegeIterator implements Iterator {
    //这里我们需要知道Department 是以怎样的方式存放
    //在这里我们假定计算机学院的学系是以数组形式存放的。如果我们不知道Department的存放方式那么我们就无法给它写迭代器
    Department[] departments;
    int position = 0;

    public ComputerCollegeIterator(Department[] departments) {
        this.departments = departments;
    }

    @Override
    public boolean hasNext() {
        if(position>departments.length || departments[position]== null){
            return false;
        }else{
            return true;
        }
    }

    @Override
    public Object next() {
        Department department = departments[position];
        position += 1;
        return department;
    }
}
