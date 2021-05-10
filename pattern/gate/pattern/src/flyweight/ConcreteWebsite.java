package flyweight;

//具体的网站产品
public class ConcreteWebsite extends WebSite {
    //网站产品的发布形式
    //内部状态，如同棋子的黑白一样。
    private String type = "";
    //构造器
    public ConcreteWebsite(String type){
        this.type = type;
    }
    @Override
    public void use(User user){
        System.out.println("网站的发布形式为："+type+ ",运行中，使用者："+user.getName());
    }
}
