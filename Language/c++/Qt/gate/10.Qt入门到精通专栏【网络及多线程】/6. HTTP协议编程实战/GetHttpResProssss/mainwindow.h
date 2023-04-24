#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>


#include <QtNetwork> // 提供编程TCP/IP客户端和服务器的类
#include <QUrl> // 提供接口使用URLs

class QNetworkAccessManager;
class QNetworkReply; // 此类是QIODevice的子类


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



    QNetworkAccessManager *mgr;
public slots:
    void replayFinishedFunc(QNetworkReply *); // 响应







private slots:
    void on_pushButton_GetData_clicked();
};
#endif // MAINWINDOW_H
