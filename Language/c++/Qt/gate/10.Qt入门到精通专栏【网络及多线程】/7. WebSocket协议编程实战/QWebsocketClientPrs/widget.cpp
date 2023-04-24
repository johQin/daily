#include "widget.h"
#include "ui_widget.h"


#include <QMessageBox>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    websocket=nullptr;
}

Widget::~Widget()
{
    delete ui;

    if(websocket)
        websocket->close();
}


void Widget::receviedMsgFunc(const QString &msg) // 接收消息
{
    QJsonDocument jsd=QJsonDocument::fromJson(msg.toUtf8().data());

    if(jsd.isNull())
    {
        ui->textEdit_MsgList->append("收到消息:"+msg);
    }
    else
    {
        QJsonObject jsobj=jsd.object();
        ui->textEdit_MsgList->append("收到来自"+jsobj["SRC"].toString()+"的消息:"+jsobj["MSG"].toString());
    }

}

void Widget::onerrorFunc(QAbstractSocket::SocketError error)
{
    ui->textEdit_MsgList->append(websocket->origin()+"出错"+websocket->errorString());
}

void Widget::on_pushButton_Connect_clicked()
{
    if(!websocket) // 实现连接与断开服务器
    {
        // 判断服务器名称是否为空
        if(ui->lineEdit_ServerName->text().trimmed().isEmpty())
        {
            QMessageBox::critical(this,"错误","服务器名称不能为空，请重新检查!",QMessageBox::Yes);
            return;
        }

        websocket=new QWebSocket(ui->lineEdit_ServerName->text().trimmed(),QWebSocketProtocol::VersionLatest,this);
        connect(websocket,&QWebSocket::connected,this,[this]
        {
            ui->textEdit_MsgList->append("已经连接上"+websocket->peerAddress().toString());
            bConnect=true;
            ui->pushButton_Connect->setText("断开服务器");
        });

        connect(websocket,&QWebSocket::connected,this,[this]
        {
            ui->textEdit_MsgList->append("已"+websocket->peerAddress().toString()+"断开连接");
            bConnect=false;
            ui->pushButton_Connect->setText("连接服务器");
        });

        connect(websocket,SIGNAL(error(QAbstractSocket::SocketError)),this,SLOT(onerrorFunc(QAbstractSocket::SocketError)));
        connect(websocket,&QWebSocket::textMessageReceived,this,&Widget::receviedMsgFunc);
    }

    if(!bConnect)
        websocket->open(QUrl(ui->lineEdit_ServerAddress->text().trimmed()));
    else
    {
        websocket->close();
        websocket->deleteLater();
        websocket=nullptr;
    }



}

void Widget::on_pushButton_SendMsg_clicked()
{
    if(!websocket)
        return;

    if(!websocket->isValid())
        return;

    // 获取发送数据信息
    QString strtext=ui->textEdit_SendData->toPlainText().trimmed();
    if(strtext.isEmpty())
        return;


    // 获取客户端名称
    QString strclient=ui->lineEdit_ClientName->text().trimmed();
    if(strclient.isEmpty())
    {
        websocket->sendTextMessage(strtext);
        ui->textEdit_MsgList->append("发送消息："+strtext);
    }
    else
    {
        QJsonObject jsobj;
        jsobj["src"]=websocket->origin();
        jsobj["dst"]=strclient;
        jsobj["msg"]=strtext;
        websocket->sendTextMessage(QString(QJsonDocument(jsobj).toJson(QJsonDocument::Compact)));
        ui->textEdit_MsgList->append("给客户端"+strclient+"发送消息:"+strtext);

    }
    ui->textEdit_SendData->clear();

}
