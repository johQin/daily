package flyweight;

public class Flyweight_src {
    public static void main(String[] args) {
        //入托Integer.valueOf(x) 的x在-127~128之间，用的就是享元模式的池，如果不在就创建

        //1. valueOf方法中，先判断值是不是在IntegerCache中，如果不在，就创建新的Integer，否则直接从池中取用返回
        //2. valueOf方法，就使用到享元模式
        //3. 通过valueOf在-127~128之间得到一个数的速度，要比new Integer快的多

        Integer x = Integer.valueOf(127);
        Integer y = new Integer(127);
        Integer z = Integer.valueOf(127);
        Integer w = new Integer(127);
        System.out.println(x.equals(y));//true
        System.out.println(x==y);
        System.out.println(x==z);//true
        System.out.println(w==x);
        System.out.println(w==y);

        Integer x1 = Integer.valueOf(200);
        Integer x2 = Integer.valueOf(200);
        System.out.println(x1 == x2);//false
        
    }
}
