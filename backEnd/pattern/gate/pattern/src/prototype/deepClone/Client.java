package prototype.deepClone;

public class Client {
    public static void main(String[] args) throws Exception{
        //方式一：重写clone()完成深拷贝
        DeepProtoType dp = new DeepProtoType();
        dp.name = "zy";
        dp.deepCloneableTarget = new DeepCloneableTarget("yuer","cute");

        DeepProtoType dp1 = (DeepProtoType) dp.clone();
        DeepProtoType dp2 = (DeepProtoType) dp.clone();
        System.out.println(dp1);
        System.out.println(dp2);

        //方式二：通过序列化对象完成深拷贝
        DeepProtoType dps = new DeepProtoType();
        dps.name = "zhangy";
        dps.deepCloneableTarget = new DeepCloneableTarget("yuer","lively");
        DeepProtoType dps1 = (DeepProtoType) dps.deepClone();
        DeepProtoType dps2 = (DeepProtoType) dps.deepClone();
        System.out.println(dps1);
        System.out.println(dps2);
    }
}
