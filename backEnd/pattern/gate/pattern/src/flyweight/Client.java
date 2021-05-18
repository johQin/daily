package flyweight;

public class Client {
    public static void main(String[] args) {
        //创建一个工厂类
        WebsiteFactory factory = new WebsiteFactory();
        //客户要一个以新闻形式发布的产品的网站
        //type为内部状态，而user为外部状态
        WebSite webSite1 = factory.getWebSiteCategory("新闻");
        webSite1.use(new User("zhangy"));
        WebSite webSite2 = factory.getWebSiteCategory("新闻");
        webSite2.use(new User("zhangyy"));
        WebSite webSite3 = factory.getWebSiteCategory("博客");
        webSite3.use(new User("zy"));
        WebSite webSite4 = factory.getWebSiteCategory("微信公众号");
        webSite4.use(new User("zyy"));
        WebSite webSite5 = factory.getWebSiteCategory("微信公众号");
        webSite5.use(new User("yuer"));
        factory.getWebsiteCount();
    }
}
