public class OCPimprove{
    public static void main(String[] args) {
        GraphEditor ge=new GraphEditor();
        ge.drawShape(new Ractangle());
        ge.drawShape(new Circle());
        ge.drawShape(new Triangle());
    }
}
//增加新功能，提供方增加了新的实现类，使用方没有任何改变
class GraphEditor{
    public void drawShape(Shape s){
        s.draw();
    }
}
abstract class Shape{
    public draw();
}
class Ractangle extends Shape{
    @Override
    public draw(){
       System.out.println("绘制矩形")
    }
}
class Circle extends Shape{
    @Override
    public draw(){
        System.out.println("绘制圆形")
     }
}
//新增三角形
class Triangle extends Shape{
    @Override
    public draw(){
        System.out.println("绘制三角形")
     }
}