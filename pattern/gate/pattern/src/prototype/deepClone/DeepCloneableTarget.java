package prototype.deepClone;

import java.io.Serializable;

//这个类是上层类中成员变量的值，用于测试深拷贝
public class DeepCloneableTarget implements Serializable, Cloneable {

    //深拷贝方法二中会用到
    private static final long serialVersionUID = 1L;

    private String cloneName;
    private String cloneClass;

    //构造器
    public DeepCloneableTarget(String cloneName, String cloneClass) {
        this.cloneName = cloneName;
        this.cloneClass = cloneClass;
    }

    //因为该类的属性，都是String , 因此我们这里使用默认的clone完成即可
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}