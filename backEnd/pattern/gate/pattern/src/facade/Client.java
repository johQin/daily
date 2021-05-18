package facade;

public class Client {
    public static void main(String[] args) {
        HomeCinemaFacade hCFacade = new HomeCinemaFacade();
        hCFacade.ready();
        hCFacade.play();
        hCFacade.pause();
        hCFacade.end();
    }
}
