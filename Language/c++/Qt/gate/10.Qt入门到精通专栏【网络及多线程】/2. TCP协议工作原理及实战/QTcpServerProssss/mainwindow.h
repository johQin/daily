#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QTcpServer> // 专门用于建立TCP连接并传输数据信息
#include <QtNetwork> // 此模块提供开发TCP/IP客户端和服务器的类

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;




    // 自定义如下
private:
    QTcpServer *tcpserver; //TCP服务器
    QTcpSocket *tcpsocket;// TCP通讯socket
    QString GetLocalIpAddress(); // 获取本机的IP地址

private slots:
    void clientconnect();
    void clientdisconnect();
    void socketreaddata();
    void newconnection();



    void on_pushButton_Start_clicked();
    void on_pushButton_Stop_clicked();
    void on_pushButton_Send_clicked();

protected:
    void closeEvent(QCloseEvent *event);


};
#endif // MAINWINDOW_H
