package Memento.theory;

public class Originator {
    private String state;
    public String getState(){
        return state;
    };
    public void setState(String state){
        this.state = state;
    }

    public Originator(String state) {
        this.state = state;
    }

    //编写一个方法，可以保存一个状态对象Memonto
    //因此编写一个方法，返回memonto
    public Memento saveStateMemonto(){
        return new Memento(state);
    }
    //通过备忘录，恢复状态
    public void getStateFromMemonto(Memento memento){
        state = memento.getState();
    }
}
