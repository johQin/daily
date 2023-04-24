#include "handlethread.h"

HandleThread::HandleThread(int socketdescriptor,QObject *parent)
    :QThread(parent),socketdescriptor(socketdescriptor)
{

}

void HandleThread::run() // 通过start()调用run()执行
{
    QTcpSocket  tcpsocket;

    // 将新创建的通信套接字描述符指定为参数socketdescriptor
    if(!tcpsocket.setSocketDescriptor(socketdescriptor))
    {
        // emit是不同窗口/类间的触发信号
        emit error(tcpsocket.error());
        return;
    }

    QByteArray blocks;

    QDataStream out(&blocks,QIODevice::WriteOnly);

    out.setVersion(QDataStream::Qt_5_12);

    tcpsocket.write(blocks);
    tcpsocket.disconnectFromHost();


}
