package Composite;

import java.util.HashMap;
import java.util.Map;

public class JDKSrc {
    public static void main(String[] args) {
        Map<Integer,String> hashMap = new HashMap<Integer,String>();
        hashMap.put(0,"西游记");

        Map<Integer,String> map = new HashMap<Integer,String>();
        map.put(1,"红楼梦");
        map.put(2,"水浒传");
        hashMap.putAll(map);
        System.out.println(hashMap);
    }


}
