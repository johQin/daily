package visitor;

import java.util.LinkedList;
import java.util.List;

//数据结构，管理很多人
public class ObjectStructure {
    //维护了一个集合
    private List<Person> persons = new LinkedList<>();
    //添加到集合
    public void attach(Person p){
        persons.add(p);
    }
    //从集合中移除
    public void detach(Person p){
        persons.remove(p);
    }
    //显示评价情况
    public void display(Action action){
        for(Person p:persons){
            p.accept(action);
        }
    }

}
