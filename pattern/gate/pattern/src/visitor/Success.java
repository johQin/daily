package visitor;

public class Success extends Action{
    @Override
    public void getManValuate(Man man) {
        System.out.println("男性观众评价该歌手演唱很成功");
    }

    @Override
    public void getWomanValuate(Woman woman) {
        System.out.println("女性观众评价该歌手演唱很成功");
    }
}
