
public class STT{
    private static STT ST=new STT();
    public static int va=50;
    public static int vb;
    private STT(){
        va-=50;
        vb+=50;
    }
    public static STT getInstance(){
        return ST;
    }
    public static void main(String[] args) {
        STT St=STT.getInstance();
        System.out.println(St.va);
        System.out.println(St.vb);

    }
}
