import java.lang.*;
public class OpenClose{
    public static void main(String[] args) {
        GraphEditor ge=new GraphEditor();
        ge.drawShape(new Ractangle());
        ge.drawShape(new Circle());
    }
}
//新增三角形画法时，
//提供方增加Triangle类
//并且使用方GraphEditor改动较多，首先在drawShaoe()增加判断条件，然后在下面还要增加drawTriangle()
//违背了ocp
class GraphEditor{
    public void drawShape(Shape s){
        if(s.m_type==1){
            drawRectangle();
        }
        else if(s.m_type==2){
            drawCircle();
        }else if(s.m_type==3){
            drawTriangle();
        }else{
            System.out.println("其他图形");
        }
    }
    public void drawRectangle(){
        System.out.println("绘制矩形");
    }
    public void drawCircle(){
        System.out.println("绘制圆形");
    }
    public void drawTriangle(){
        System.out.println("绘制圆形");
    }
}
class Shape{
    int m_type;
}
class Ractangle extends Shape{
    public Ractangle(){
        super.m_type=1;
    }
}
class Circle extends Shape{
    public Circle(){
        super.m_type=2;
    }
}
//新增三角形
class Triangle extends Shape{
    public Triangle(){
        super.m_type=3;
    }
}