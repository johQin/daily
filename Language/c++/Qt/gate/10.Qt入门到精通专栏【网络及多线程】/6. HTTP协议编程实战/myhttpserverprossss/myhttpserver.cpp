#include "myhttpserver.h"

#include <QtDebug>

myhttpserver::myhttpserver(QObject *parent) : QObject(parent)
{
    ser=new QTcpServer(this);

    // 信号与槽函数连接
    connect(ser, &QTcpServer::newConnection, this, &myhttpserver::connection);

    if(!ser->listen(QHostAddress::Any,8088))
    {
        qDebug()<<"\n致命错误：Web服务器没有启动，请重新检查!"<<endl;
    }
    else
    {
        qDebug()<<"\n正常启动：Web服务器端口:8088，等待客户端连接......"<<endl;;
    }
}

myhttpserver::~myhttpserver()
{
    socket->close();
}


void myhttpserver::connection() // 连接
{
    socket=ser->nextPendingConnection();

    while (!(socket->waitForReadyRead(100)));

    char webdata[1000];
    int sv=socket->read(webdata,1000);
    qDebug()<<"正常运行：从浏览器读取数据信息......"<<QString(webdata);

    socket->write("HTTP/1.1 200 OK\r\n");
    socket->write("Content-Type: text/html\r\n");
    socket->write("Connection: close\r\n");

    socket->write("Refresh: 3\r\n\r\n"); // 每秒刷新Web浏览器

    socket->write("<!DOCTYPE>"
                  "<html>"
                  "<header>"
                  "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"/>"
                  "<title>HttpServer</title>"
                  "</header>"
                  "<body>客户端已经连接HttpSever服务器秒数为：");

    QByteArray bytesary;
    static qint16 icount;    // 用于在浏览器上显示的统计访问数字
    bytesary.setNum(icount++);
    socket->write(bytesary);
    socket->write("</html>");

    socket->flush();
    connect(socket, &QTcpSocket::disconnected, socket, &QTcpSocket::deleteLater);
    socket->disconnectFromHost();

}
