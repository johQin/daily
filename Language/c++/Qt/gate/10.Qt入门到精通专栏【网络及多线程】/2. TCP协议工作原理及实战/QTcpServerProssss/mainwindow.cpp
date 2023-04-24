#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString strip=GetLocalIpAddress();
    // QMessageBox::information(this,"数据",strip,QMessageBox::Yes);

    ui->comboBoxIp->addItem(strip);


    tcpserver=new QTcpServer(this);

    connect(tcpserver,SIGNAL(newConnection()),this,SLOT(newconnection()));




}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_Start_clicked()
{
    QString ip=ui->comboBoxIp->currentText();
    quint16 port=ui->spinBoxPort->value();

    QHostAddress address(ip);
    tcpserver->listen(address,port);

    ui->plainTextEdit_DispMsg->appendPlainText("$$$$$$$$$$开始监听$$$$$$$$$$");
    ui->plainTextEdit_DispMsg->appendPlainText("$$$$$$$$$$服务器地址$$$$$$$$$$："+
                                               tcpserver->serverAddress().toString());
    ui->plainTextEdit_DispMsg->appendPlainText("$$$$$$$$$$服务器端口$$$$$$$$$$："+
                                               QString::number(tcpserver->serverPort()));
    ui->pushButton_Start->setEnabled(false);
    ui->pushButton_Stop->setEnabled(true);

}

void MainWindow::on_pushButton_Stop_clicked()
{
    if(tcpserver->isListening())
    {
        tcpserver->close();
        ui->pushButton_Start->setEnabled(true);
        ui->pushButton_Stop->setEnabled(false);
    }

}

void MainWindow::on_pushButton_Send_clicked()
{
    QString strmsg=ui->lineEdit_InputMsg->text();
    ui->plainTextEdit_DispMsg->appendPlainText("[out]:"+strmsg);

    ui->lineEdit_InputMsg->clear();

    QByteArray str=strmsg.toUtf8();
    str.append("\n");
    tcpsocket->write(str);
}


QString MainWindow::GetLocalIpAddress() // 获取本机的IP地址
{
    QString hostname=QHostInfo::localHostName();
    QHostInfo hostinfo=QHostInfo::fromName(hostname);

    QString localip="";

    QList<QHostAddress> addresslist=hostinfo.addresses();

    if(!addresslist.isEmpty())
    {
        for (int i=0;i<addresslist.count();i++)
        {
            QHostAddress addrhost=addresslist.at(i);
            if(QAbstractSocket::IPv4Protocol==addrhost.protocol())
            {
                localip=addrhost.toString();
                break;
            }

        }
    }

    return localip;
}

void MainWindow::clientconnect()
{
    // 客户端连接
    ui->plainTextEdit_DispMsg->appendPlainText("**********客户端socket连接**********");
    ui->plainTextEdit_DispMsg->appendPlainText("**********peer address:"+
                                               tcpsocket->peerAddress().toString());
    ui->plainTextEdit_DispMsg->appendPlainText("**********peer port:"+
                                               QString::number(tcpsocket->peerPort()));

}

void MainWindow::clientdisconnect()
{
    // 客户端断开连接
    ui->plainTextEdit_DispMsg->appendPlainText("**********客户端socket断开连接**********");
    tcpsocket->deleteLater();

}

void MainWindow::socketreaddata()
{
    // 读取数据
    while(tcpsocket->canReadLine())
        ui->plainTextEdit_DispMsg->appendPlainText("[in]"+tcpsocket->readLine());

}

void MainWindow::newconnection()
{
    tcpsocket=tcpserver->nextPendingConnection();

    connect(tcpsocket,SIGNAL(connected()),this,SLOT(clientconnect()));
    clientconnect();

    connect(tcpsocket,SIGNAL(disconnected()),this,SLOT(clientdisconnect()));

    connect(tcpsocket,SIGNAL(readyRead()),this,SLOT(socketreaddata()));

    connect(tcpsocket,SIGNAL(stateChanged(QAbstractSocket::SocketState)),
            this,SLOT(OnSocketStateChanged(QAbstractSocket::SocketState)));


}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if(tcpserver->isListening())
        tcpserver->close();

    event->accept();
}
