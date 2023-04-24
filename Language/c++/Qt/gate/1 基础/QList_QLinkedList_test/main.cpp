#include <QCoreApplication>
#include<QDebug>
#include<QLinkedList>
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QList<int> qlist;
    for(int i=0; i<10; ++i){
        qlist.insert(qlist.end(),i+10);     //插入元素
    }       //(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)

    // QList<int>::iterator 读写迭代器
    QList<int>::iterator x;
    for(x=qlist.begin();x!=qlist.end();x++){
        *x = *x *10 + 6;
    }       //(106, 116, 126, 136, 146, 156, 166, 176, 186, 196)
    // 还有const_iterator只读迭代器，开始需要用constBegin(),结尾要用constEnd()

    //添加元素
    qlist.append(888);
    //查询元素
    qlist.at(3);// 通过索引3，第四个元素，查询元素值，136
    qlist.contains(136);
    //修改qlist列表里的值
    qlist.replace(5,999);//通过索引5，第6个元素，修改为999
    //删除元素
    qlist.removeAt(0);
    qlist.removeFirst();
    qlist.removeAt(6);


    qDebug()<<qlist;

    //使用QLinkedList需要包含头文件。
    QLinkedList<QString> qAllMonth;
    for(int i =1; i<=12;++i){
        qAllMonth<<QString("%1%2").arg("Month:").arg(i);
    }

    QLinkedList<QString>::iterator itrw = qAllMonth.begin();
    for(;itrw!=qAllMonth.end();++itrw){
        qDebug()<<*itrw;
    }
    return a.exec();
}
