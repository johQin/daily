package visitor;

public class Fail extends Action{
    @Override
    public void getManValuate(Man man) {
        System.out.println("男性观众评价该歌手演唱很失败");
    }

    @Override
    public void getWomanValuate(Woman woman) {
        System.out.println("女性观众评价该歌手演唱很失败");
    }
}
