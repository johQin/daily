#include "mainwindow.h"
#include<QVariant>
#include<QDebug>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 1. 一般类型的使用
    QVariant qv1("ABC");
    qDebug()<<"qv1:"<<qv1.toString();

    // 2. QVariant与QMap结合使用
    QMap<QString,QVariant> qmap;
    qmap["int"] = 2000;
    qmap["double"] = 90.99;
    qmap["string"] = "Good";
    qmap["color"] = QColor(255,255,0);

    //输出：转换函数来处理
    qDebug()<<qmap["int"]<<" "<<qmap["int"].toInt();            //QVariant(int, 2000)   2000
    qDebug()<<qmap["double"]<<" "<<qmap["double"].toDouble();   //QVariant(double, 90.99)   90.99
    qDebug()<<qmap["string"]<<" "<<qmap["string"].toString();   //QVariant(QString, "Good")   "Good"
    qDebug()<<qmap["color"]<<" "<<qmap["color"].value<QColor>();//QVariant(QColor, QColor(ARGB 1, 1, 1, 0))   QColor(ARGB 1, 1, 1, 0)

    // 3. QVariant与QStringList
    QStringList qsl;
    qsl<<"A"<<"B"<<"C"<<"D";
    QVariant qvsl(qsl);
    if(qvsl.type() == QVariant::StringList){
        QStringList qlist = qvsl.toStringList();
        for(int i=0; i<qlist.size();++i){
            qDebug()<<qlist.at(i);
        }   // "A"  "B"  "C"  "D"
    }

    // 4.  QVariant与自定义结构体Student
    // 在.h文件中，定义及声明为Q_DECLARE_METATYPE
    //struct Student{
    //    int no;
    //    QString name;
    //};
    //Q_DECLARE_METATYPE(Student)
    Student stu;
    stu.no = 12345;
    stu.name = "qkk";

    QVariant qstu = QVariant::fromValue(stu);
    //判断是否可以转换原始对象
    if(qstu.canConvert<Student>()){
        //获取原始对象方式一
        Student temp = qstu.value<Student>();       //student: { no:  12345  name:  "qkk" }
        //获取原始对象方式二
        Student qtemp = qvariant_cast<Student>(qstu);//student: { no:  12345  name:  "qkk" }
        qDebug()<<"student: {"<<"no: "<<temp.no<<" name: "<<temp.name<<"}";
    }
}

MainWindow::~MainWindow()
{
}

