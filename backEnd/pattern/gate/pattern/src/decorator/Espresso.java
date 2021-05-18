package decorator;

public class Espresso extends Coffee{
    public Espresso(){
        setPrice(6.0f);
        setDes("Espresso"+getPrice());
    }
}
