#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // WS（服务器以非安全模式运行）
    // WSS（服务器以安全械运行）
    websocketserver=new QWebSocketServer(QStringLiteral("testServer"),
                                         QWebSocketServer::NonSecureMode,this);

    connect(websocketserver,&QWebSocketServer::newConnection,this,&Widget::getnewconnect);

    websocketserver->listen(QHostAddress::Any,8899);
}

Widget::~Widget()
{
    delete ui;

    for(auto socket:websocketlist)
    {
        socket->close();
    }
    websocketserver->close();
}


void Widget::on_pushButton_SendData_clicked()
{
    QString strtext=ui->textEdit_SendData->toPlainText().trimmed();
    if(strtext.isEmpty())
        return;

    if(ui->radioButton_SendAll->isChecked()) // 群发客户端
    {
        if(websocketlist.size()==0)
            return;
        for(auto socket:websocketlist)
        {
            socket->sendTextMessage(strtext);
        }
        ui->textEdit_MsgList->append("服务器给所有连接发送："+strtext);
    }
    else // 私发客户端
    {
        if(!ui->listWidget_Client->currentItem())
        {
            return ;
        }

        QString strcurrent=ui->listWidget_Client->currentItem()->text();

        QWebSocket *websocket=nullptr;

        for(auto socket:websocketlist)
        {
            if(socket->origin()==strcurrent)
            {
                websocket=socket;
                break;
            }
        }

        if(websocket)
        {
            websocket->sendTextMessage(strcurrent);
            ui->textEdit_MsgList->append("服务端给["+websocket->origin()+"]发送--->"+strtext);
        }
    }

    ui->textEdit_SendData->clear();
}

void Widget::getnewconnect() // 获取新连接
{
    // 如果服务器有挂起的连接，返回true
    if(websocketserver->hasPendingConnections())
    {
        QWebSocket *websocket=websocketserver->nextPendingConnection();

        ui->textEdit_MsgList->append(websocket->origin()+"客户端已连接到服务器");
        websocketlist<<websocket;

        QListWidgetItem *item=new QListWidgetItem;
        item->setText(websocket->origin());
        ui->listWidget_Client->addItem(item); // 将连接的客户端添加到客户端列表控件

        connect(websocket,&QWebSocket::disconnected,this,[websocket,this]
        {
            ui->textEdit_MsgList->append(websocket->origin()+"客户端已断开服务器");
            websocketlist.removeOne(websocket);
            for(int i=0;i<ui->listWidget_Client->count();i++)
            {
                QListWidgetItem *item=ui->listWidget_Client->item(i);

                if(item->text()==websocket->origin())
                {
                    ui->listWidget_Client->removeItemWidget(item);
                    delete item;
                    break;
                }
            }
            websocket->deleteLater();
        });

        connect(websocket,&QWebSocket::textMessageReceived,this,&Widget::receiveMsg);
        connect(websocket,SIGNAL(error(QAbstractSocket::SocketError)),this,SLOT(onerrorFunc(QAbstractSocket::SocketError)));
    }

}

void Widget::receiveMsg(const QString &msg)
{
    QJsonDocument jd=QJsonDocument::fromJson(msg.toLatin1().data());

    if(jd.isNull())
    {
        QWebSocket *websocket=qobject_cast<QWebSocket*>(sender());
        ui->textEdit_MsgList->append("收到客户端消息["+websocket->origin()+"]--->"+msg);
    }
    else
    {
        QJsonObject jdo=jd.object();
        QString dst=jdo["dst:"].toString();
        for (auto socket:websocketlist)
        {
            if(dst==socket->origin())
                socket->sendTextMessage(msg);
        }
    }
}


void Widget::onerrorFunc(QAbstractSocket::SocketError error)
{
    QWebSocket *websocket=qobject_cast<QWebSocket*>(sender());

    ui->textEdit_MsgList->append(websocket->origin()+"出错"+websocket->errorString());
}















