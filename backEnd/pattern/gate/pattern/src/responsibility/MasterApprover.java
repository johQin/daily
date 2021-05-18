package responsibility;

public class MasterApprover extends Approver{
    public MasterApprover(String name) {
        super(name);
    }

    @Override
    public void processRequest(PuchaseRequest puchaseRequest) {
        if(puchaseRequest.getPrice()>15000){
            System.out.println("请求编号id="+puchaseRequest.getId()+"被"+this.name+"处理");
        }else{
            approver.processRequest(puchaseRequest);
        }
    }
}