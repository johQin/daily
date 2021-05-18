package visitor;

public abstract class Action {
    //得到男性观众的评价
    public abstract void getManValuate(Man man);
    //得到女性观众的评价
    public abstract void getWomanValuate(Woman woman);
}
