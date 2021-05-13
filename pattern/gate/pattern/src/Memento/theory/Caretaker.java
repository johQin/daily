package Memento.theory;

import java.util.ArrayList;
import java.util.List;

public class Caretaker {
    //聚合多个memento对象
    private List<Memento> mementos = new ArrayList<Memento>();
    public void addMemento(Memento memento){
        mementos.add(memento);
    }
    //获取到第index个originator的备忘录对象Memento
    public Memento get(int index){
        return mementos.get(index);
    }
}
