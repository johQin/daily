package proxy.staticproxy;

//代理
public class TeacherDaoProxy implements ITeacherDao{
    private ITeacherDao target;
    //构造器
    public TeacherDaoProxy(ITeacherDao target){
        this.target = target;
    }
    @Override
    public void teach() {
        System.out.println("代理开始");
        target.teach();
        System.out.println("代理结束");
    }
}
