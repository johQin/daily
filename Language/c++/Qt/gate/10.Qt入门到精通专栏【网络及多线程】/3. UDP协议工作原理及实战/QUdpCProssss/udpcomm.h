#ifndef UDPCOMM_H
#define UDPCOMM_H

#include <QMainWindow>


#include <QUdpSocket> // 用于发送和接收UDP数据报
#include <QtNetwork>





QT_BEGIN_NAMESPACE
namespace Ui { class udpComm; }
QT_END_NAMESPACE

class udpComm : public QMainWindow
{
    Q_OBJECT

public:
    udpComm(QWidget *parent = nullptr);
    ~udpComm();

private slots:
    void on_pushButton_start_clicked();

    void on_pushButton_stop_clicked();

    void on_pushButton_sendmsg_clicked();

    void on_pushButton_broadcastmsg_clicked();

private:
    Ui::udpComm *ui;




    // 自定义函数获取本机的IP地址
public:
    QUdpSocket *udpsocket;

    QString GetLocalIpAddress(); // 获取本机IP地址

    // 自己定义槽
private slots:
    void  SocketReadyReadData(); // 读取socket传输数据信息















};
#endif // UDPCOMM_H
