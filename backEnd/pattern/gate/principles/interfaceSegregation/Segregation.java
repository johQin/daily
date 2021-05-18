public class Segregation{
    public static void main(String[] args){
        A a=new A();
        a.depend1(new B());
        //interface1的实现类B向interface1自动转型，
        //多态：java执行方法时，方法的执行是动态绑定的，方法总是执行该变量实际所指向对象的方法（只针对非静态的方法）。
        ASG asg = new ASG();
        asg.depend1(new BSG());
        asg.depend2(new BSG());
        asg.depend3(new BSG());
        CSG csg = new CSG();
        csg.depend1(new DSG());
        csg.depend2(new DSG());
        csg.depend3(new DSG());
    }
}
class A{//A通过接口interface1 依赖使用实现类B,但是只会用到1,2,3方法
    public void depend1(Interface1 i){
        i.operation1();
    }
    public void depend2(Interface1 i){
        i.operation2();
    }
    public void depend3(Interface1 i){
        i.operation3();
    }
}
class C{//C通过接口interface1 依赖使用实现类D,但是只会用到1,4,5方法
    public void depend1(Interface1 i){
        i.operation1();
    }
    public void depend4(Interface1 i){
        i.operation4();
    }
    public void depend5(Interface1 i){
        i.operation5();
    }
}
interface Interface1{
    void operation1();
    void operation2();
    void operation3();
    void operation4();
    void operation5();
}
class B implements Interface1{
    public void operation1(){
        System.out.println(" 类B 实现了接口1的方法1");
    }
    public void operation2(){
        System.out.println(" 类B 实现了接口1的方法2");
    }
    public void operation3(){
        System.out.println(" 类B 实现了接口1的方法3");
    }
    public void operation4(){
        System.out.println(" 类B 实现了接口1的方法4");
    }
    public void operation5(){
        System.out.println(" 类B 实现了接口1的方法5");
    }
}
class D implements Interface1{
    public void operation1(){
        System.out.println(" 类D 实现了接口1的方法1");
    }
    public void operation2(){
        System.out.println(" 类D 实现了接口1的方法2");
    }
    public void operation3(){
        System.out.println(" 类D 实现了接口1的方法3");
    }
    public void operation4(){
        System.out.println(" 类D 实现了接口1的方法4");
    }
    public void operation5(){
        System.out.println(" 类D 实现了接口1的方法5");
    }
}


//下面符合接口隔离原则
class ASG{//ASG通过接口Interface1SG和Interface2SG 依赖使用实现类BSG,故BSG只需实现三个方法，也就是ASG依赖的三个方法
    public void depend1(Interface1SG i){
        i.operation1();
    }
    public void depend2(Interface2SG i){
        i.operation2();
    }
    public void depend3(Interface2SG i){
        i.operation3();
    }
}
class CSG{//CSG通过接口Interface1SG和Interface3SG 依赖使用实现类DSG,故DSG只需实现三个方法，也就是DSG依赖的三个方法
    public void depend1(Interface1SG i){
        i.operation1();
    }
    public void depend2(Interface3SG i){
        i.operation4();
    }
    public void depend3(Interface3SG i){
        i.operation5();
    }
}
interface Interface1SG{
    void operation1();
}
interface Interface2SG{
    void operation2();
    void operation3();
}
interface Interface3SG{
    void operation4();
    void operation5();
}
class BSG implements Interface1SG,Interface2SG{//BSG实现了Interface1SG,Interface2SG
    public void operation1(){
        System.out.println(" 类BSG 实现了接口1SG的方法1");
    }
    public void operation2(){
        System.out.println(" 类BSG 实现了接口2SG的方法2");
    }
    public void operation3(){
        System.out.println(" 类BSG 实现了接口2SG的方法3");
    }
}
class DSG implements Interface1SG,Interface3SG{//DSG实现了Interface1SG,Interface3SG
    public void operation1(){
        System.out.println(" 类DSG 实现了接口1SG的方法1");
    }
    public void operation4(){
        System.out.println(" 类DSG 实现了接口2SG的方法4");
    }
    public void operation5(){
        System.out.println(" 类DSG 实现了接口2SG的方法5");
    }
}
