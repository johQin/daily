package Composite;

public class Client {
    public static void main(String[] args) {
        OrganizationComponent university = new University("清华大学","世界一流大学");

        OrganizationComponent computerCollege = new College("计算机学院","中国前10");
        OrganizationComponent informationCollege = new College("信息工程学院","中国前20");

        computerCollege.add(new Department("软件工程", " 软件工程优秀"));
        computerCollege.add(new Department("网络工程", " 网络工程特特别优秀 "));
        computerCollege.add(new Department("计算机科学与技术", " 计算机科学与技术是老牌的专业 "));

        informationCollege.add(new Department("通信工程", " 通信工程不好学 "));
        informationCollege.add(new Department("信息工程", " 信息工程好学 "));

        //将学院加入到 学校
        university.add(computerCollege);
        university.add(informationCollege);

        university.print();
        informationCollege.print();


    }
}
