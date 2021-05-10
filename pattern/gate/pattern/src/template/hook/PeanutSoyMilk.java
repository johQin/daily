package template.hook;

public class PeanutSoyMilk extends SoyMilk {
    @Override
    void addIngredient() {
        System.out.println("加入花生，可以使豆浆更加细腻浓郁");
    }
}
