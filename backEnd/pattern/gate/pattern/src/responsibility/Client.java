package responsibility;

public class Client {
    public static void main(String[] args) {

        //创建请求
        PuchaseRequest puchaseRequest = new PuchaseRequest(1,30000,1);

        //创建各级负责人
        DepartmentApprover departmentApprover = new DepartmentApprover("系主任");
        CollegeApprover collegeApprover = new CollegeApprover("院长");
        ViceMasterApprover viceMasterApprover = new ViceMasterApprover("副校长");
        MasterApprover masterApprover = new MasterApprover("校长");

        //将各个级别的下一级，设置好，并形成环状（使职责链的起点可以为任何级别的负责人）
        departmentApprover.setApprover(collegeApprover);
        collegeApprover.setApprover(viceMasterApprover);
        viceMasterApprover.setApprover(masterApprover);
        masterApprover.setApprover(departmentApprover);

        //正规处理，是从最级别的负责人开始处理
        departmentApprover.processRequest(puchaseRequest);

    }
}
