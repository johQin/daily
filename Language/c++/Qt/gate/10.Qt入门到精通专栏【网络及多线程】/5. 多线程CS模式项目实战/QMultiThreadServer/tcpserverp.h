#ifndef TCPSERVERP_H
#define TCPSERVERP_H

#include <QTcpServer>
class Dialog; // 服务器端对话框的向前声明的类

#include "handlethread.h"
#include "dialog.h"

class tcpserverp : public QTcpServer // 第二步
{
    Q_OBJECT
public:
    tcpserverp(QObject *parent=0);


protected:
    // 当有新连接的时候会自动调用此函数
    void incomingConnection(int socketdescriptor);

    Dialog *dlgs;

};

#endif // TCPSERVERP_H
