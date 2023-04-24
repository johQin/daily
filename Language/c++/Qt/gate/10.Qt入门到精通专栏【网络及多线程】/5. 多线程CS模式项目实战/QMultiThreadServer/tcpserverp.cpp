#include "tcpserverp.h"

tcpserverp::tcpserverp(QObject *parent)
    :QTcpServer(parent)
{
    dlgs=(Dialog*)parent;

}

void tcpserverp::incomingConnection(int socketdescriptor)
{
    HandleThread *thread=new HandleThread(socketdescriptor,0);

    // 此处用于处理对话框显示统计访问次数信息
    connect(thread,SIGNAL(finished()),dlgs,SLOT(slotsdispFunc()));

    connect(thread,SIGNAL(finished()),thread,SLOT(deleteLater()),Qt::DirectConnection);
    thread->start(); // 通过执行这条语句来调用run()函数

}
