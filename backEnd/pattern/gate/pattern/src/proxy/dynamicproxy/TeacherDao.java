package proxy.dynamicproxy;

public class TeacherDao implements ITeacherDao {
    @Override
    public void teach() {
        System.out.println("老师授课");
    }

    @Override
    public void checkHomework(Integer number) {
        System.out.println("老师批改作业,学号为："+number);
    }
}
