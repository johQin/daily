package prototype.deepClone;

import java.io.*;

public class DeepProtoType implements Serializable,Cloneable {
    public String name;
    public DeepCloneableTarget deepCloneableTarget;
    public DeepProtoType(){
        super();
    }

    @Override
    public String toString() {
        return "DeepProtoType{" +
                "name='" + name + '\'' +
                ", deepCloneableTarget=" + deepCloneableTarget.hashCode() +
                '}';
    }

    //深拷贝
    // 方式一 通过重写clone()方法来实现
    @Override
    protected Object clone() throws CloneNotSupportedException {
        //对基本类型的数据进行拷贝
        DeepProtoType deep = (DeepProtoType) super.clone();
        //对引用类型的数据进行拷贝，如果deepCloneableTarget底层还有引用类型的对象，那么更为复杂。
        //如果成员变量中还有其他的引用类型的对象，那么深拷贝也会变得复杂。
        deep.deepCloneableTarget = (DeepCloneableTarget) deepCloneableTarget.clone();
        return deep;
    }

    //方式二 通过对象序列化实现深拷贝（推荐）
    public Object deepClone(){
        //创建流对象，初始化
        //输出流
        ByteArrayOutputStream bos = null;
        ObjectOutputStream oos = null;
        //输入流
        ByteArrayInputStream bis = null;
        ObjectInputStream ois = null;

        try{
            //序列化
            bos = new ByteArrayOutputStream();
            oos = new ObjectOutputStream(bos);
            oos.writeObject(this);//当前这个对象以对象流的方式输出

            //反序列化
            bis = new ByteArrayInputStream(bos.toByteArray());
            ois = new ObjectInputStream(bis);
            DeepProtoType copyObj = (DeepProtoType) ois.readObject();

            return copyObj;
        }catch(Exception e){
            System.out.println(e.getMessage());
            return null;
        }finally{
            //关闭流
            try{
                bos.close();
                oos.close();
                bis.close();
                ois.close();
            }catch(Exception e){
                System.out.println(e.getMessage());
            }

        }

    }
}
