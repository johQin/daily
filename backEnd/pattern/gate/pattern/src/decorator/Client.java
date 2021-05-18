package decorator;

public class Client {
    public static void main(String[] args) {
        //1.先点一份Espresso
        Drink order = new Espresso();
        System.out.println("订单描述："+order.getDes()+"，订单费用："+ order.cost());

        //2.加牛奶
        order =new Milk(order);
        System.out.println("订单描述："+order.getDes()+"，订单费用："+ order.cost());

        //3.加巧克力
        order = new Chocolate(order);
        System.out.println("订单描述："+order.getDes()+"，订单费用："+ order.cost());

        //4.再加巧克力
        order = new Chocolate(order);
        System.out.println("订单描述："+order.getDes()+"，订单费用："+ order.cost());
    }
}
