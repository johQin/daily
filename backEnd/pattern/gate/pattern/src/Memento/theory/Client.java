package Memento.theory;

public class Client {
    public static void main(String[] args) {
        Originator originator = new Originator("small");
        Caretaker caretaker = new Caretaker();
        //保存当前originator的状态
        caretaker.addMemento(originator.saveStateMemonto());
        //状态变化
        originator.setState("medium");
        //保存新的状态
        caretaker.addMemento(originator.saveStateMemonto());

        originator.setState("big");
        caretaker.addMemento(originator.saveStateMemonto());

        //当前状态
        System.out.println("当前状态：" + originator.getState());

        //恢复状态到初始状态
        originator.getStateFromMemonto(caretaker.get(0));

        System.out.println("当前状态：" + originator.getState());

    }
}
