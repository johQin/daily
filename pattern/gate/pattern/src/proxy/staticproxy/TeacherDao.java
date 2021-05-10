package proxy.staticproxy;

//被代理
public class TeacherDao implements ITeacherDao{
    @Override
    public void teach() {
        System.out.println("老师授课");
    }
}
