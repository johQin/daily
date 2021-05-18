package Iterator;

import java.util.Iterator;
import java.util.List;

public class InfoCollegeIterator implements Iterator {
    //在这里我们假设信息工程学院的学系是以List的形式存放的
    List<Department> departmentList;
    int index = -1;

    public InfoCollegeIterator(List<Department> departmentList) {
        this.departmentList = departmentList;
    }

    @Override
    public boolean hasNext() {
        if(index>=departmentList.size()-1){
            return false;
        }else{
            index += 1;
            return true;
        }
    }

    @Override
    public Object next() {

        return departmentList.get(index);
    }
}
