package template;

public abstract class SoyMilk {
    //模板方法，make，模板方法可以指定为final，不让子类覆盖
    final void make(){
        select();
        soak();
        addIngredient();
        beat();
    }
    void select(){
        System.out.println("第一步：选材，好的原料决定了口感");
    }
    abstract void addIngredient();

    void soak(){
        System.out.println("材料需要浸泡20min");
    }
    void beat(){
        System.out.println("请耐心等候，机器正在打磨");
    }
}
