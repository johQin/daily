package responsibility;

public class CollegeApprover extends Approver {
    public CollegeApprover(String name) {
        super(name);
    }

    @Override
    public void processRequest(PuchaseRequest puchaseRequest) {
        if (puchaseRequest.getPrice()>5000 && puchaseRequest.getPrice() <= 10000) {
            System.out.println("请求编号id=" + puchaseRequest.getId() + "被" + this.name + "处理");
        } else {
            approver.processRequest(puchaseRequest);
        }
    }
}
