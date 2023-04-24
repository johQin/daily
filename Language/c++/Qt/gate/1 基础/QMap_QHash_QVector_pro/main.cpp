#include <QCoreApplication>
#include<QDebug>
#include<iostream>
using namespace std;
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QVector<int> qvr;
    //添加方式1
    qvr<<10;
    qvr<<20;
    //添加方式2
    qvr.append(30);
    qvr.append(40);
    qDebug()<<qvr;  //QVector(100, 200, 300, 400)

    qvr.count();    //获取元素的个数，4
    qvr<<50;
    qvr<<60;
    qvr.remove(0);   //删除第一个元素, 删除10
    qvr.remove(1,3); //从第二个元素后，删除3个元素（左开右闭），删除30,40,50

    // 遍历
    for(int i = 0;i<qvr.count();i++){
        qDebug()<<qvr[i];
    }

    //是否包含某个元素
    qvr.contains(60);   //true
    return a.exec();
}
