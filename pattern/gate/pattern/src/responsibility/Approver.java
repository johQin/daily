package responsibility;

public abstract class Approver {
    Approver approver;//下一个处理者
    String name;//负责人姓名

    public Approver(String name) {
        this.name = name;
    }

    public void setApprover(Approver approver) {
        this.approver = approver;
    }
    //处理审批请求的方法，得到一个请求，处理是子类完成，因此该方法抽象
    public abstract void processRequest(PuchaseRequest puchaseRequest);


}
