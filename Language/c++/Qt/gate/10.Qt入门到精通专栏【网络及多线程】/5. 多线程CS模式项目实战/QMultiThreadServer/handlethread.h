#ifndef HANDLETHREAD_H
#define HANDLETHREAD_H

#include <QThread>
#include <QTcpSocket> // 建立TCP连接并传输数据流
#include <QtNetwork> // TCP/IP客户端和服务器的类
#include <QByteArray> // 字节数组 TCP、UDP发送和接收数据都是采用QByteArray
#include <QDataStream>



class HandleThread : public QThread // 第一步
{
    Q_OBJECT
public:
    HandleThread(int socketdescriptor,QObject *parent=0);


    void run();
signals:
    void error(QTcpSocket::SocketError socketerror);

private:
    int socketdescriptor;


};

#endif // HANDLETHREAD_H
