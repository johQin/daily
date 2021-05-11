package Iterator;

import java.util.Iterator;

public interface College {
    public String getName();
    public void addDepartment(String name,String desc);
    //返回一个迭代器，用于遍历
    public Iterator createIterator();
}
