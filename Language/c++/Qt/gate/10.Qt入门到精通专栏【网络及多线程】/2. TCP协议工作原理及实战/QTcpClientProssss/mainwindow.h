#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>


#include <QTcpSocket>
#include <QHostAddress>
#include <QHostInfo>

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


private:
    QTcpSocket *tcpclient; // 客户端tcpclient
    QString getlocalip(); // 获取本机IP地址

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void connectfunc();
    void disconnectfunc();
    void socketreaddata();





    void on_pushButton_Connect_clicked();
    void on_pushButton_Send_clicked();
    void on_pushButton_Disconnect_clicked();
};
#endif // MAINWINDOW_H
