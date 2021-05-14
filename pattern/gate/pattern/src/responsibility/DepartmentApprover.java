package responsibility;

public class DepartmentApprover extends Approver{
    public DepartmentApprover(String name) {
        super(name);
    }

    @Override
    public void processRequest(PuchaseRequest puchaseRequest) {
        if(puchaseRequest.getPrice()>0 && puchaseRequest.getPrice()<=5000){
            System.out.println("请求编号id="+puchaseRequest.getId()+"被"+this.name+"处理");
        }else if(puchaseRequest.getPrice()<=0){
            System.out.println("请求购买任务出错");
        }else{
            approver.processRequest(puchaseRequest);
        }
    }
}
