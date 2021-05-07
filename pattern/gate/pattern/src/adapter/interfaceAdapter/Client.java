package adapter.interfaceAdapter;

public class Client {
    public static void main(String[] args) {
        AbsAdapter absAdapter = new AbsAdapter() {
            @Override
            public void m2() {
                System.out.println("我只需要对我使用到的m2方法进行真正的实现");
            }
        };
        absAdapter.m2();
    }
}
