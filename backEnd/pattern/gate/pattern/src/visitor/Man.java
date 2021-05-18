package visitor;

//说明：
//1. 这里我们使用到一个双分派，即首先在客户端程序中，将具体状态作为参数传递到Man中（通过accept方法）
//2. 然后Man类调用作为参数的“具体方法”中方法getManValuate，同时将自己（this）作为参数
// 传入，完成第二次分派
public class Man extends Person{
    @Override
    public void accept(Action action) {
        action.getManValuate(this);
    }
}
