package flyweight;

import java.util.HashMap;

//网站工厂类，根据需要返回一个网站
public class WebsiteFactory {
    //集合，充当池的作用
    private HashMap<String,ConcreteWebsite> pool = new HashMap<>();

    //根据网站的类型，返回一个网站，如果没有就创建一个网站，并放入到池中，一并返回
    public WebSite getWebSiteCategory(String type){
        //如果没有就创建
        if(!pool.containsKey(type)){
            pool.put(type,new ConcreteWebsite(type));
        }
        return (WebSite) pool.get(type);
    }
    //获取网站分类的总数（池中有多少个网站类型）
    public Integer getWebsiteCount(){
        Integer size = pool.size();
        System.out.println("当前网站的运行总数:"+ size+"个");
        return size;
    }

}
