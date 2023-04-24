#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

#include <QAbstractSocket>
#include <QWebSocket>
#include <QJsonDocument>
#include <QJsonObject>



QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private:
    Ui::Widget *ui;

    class QWebSocket *websocket; // 套接字
    void receviedMsgFunc(const QString &msg); // 接收消息
    bool bConnect=false; // 判断连接与断开

private slots:
    void onerrorFunc(QAbstractSocket::SocketError error);
















    void on_pushButton_Connect_clicked();
    void on_pushButton_SendMsg_clicked();
};
#endif // WIDGET_H
