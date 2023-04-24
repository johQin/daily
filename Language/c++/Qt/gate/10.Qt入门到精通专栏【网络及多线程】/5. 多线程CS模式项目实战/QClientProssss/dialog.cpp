#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::Dialog)
{
    ui->setupUi(this);

    connect(ui->pushButton_Request,SIGNAL(clicked()),this,SLOT(opeexec()));
    connect(ui->pushButton_Exit,SIGNAL(clicked()),this,SLOT(close()));

    tcpsocket=new QTcpSocket(this);
    connect(tcpsocket,SIGNAL(error(QAbstractSocket::SocketError)),this,SLOT(dispError(QAbstractSocket::SocketError)));
    ui->lineEdit_ServerPort->setFocus();


}

Dialog::~Dialog()
{
    delete ui;
}


void Dialog::on_pushButton_Request_clicked()
{
    ui->pushButton_Request->setEnabled(!ui->lineEdit_ServerName->text().isEmpty() && !ui->lineEdit_ServerPort->text().isEmpty());


}

void Dialog::dispError(QAbstractSocket::SocketError socketerror)
{
    switch (socketerror)
    {
    case QAbstractSocket::RemoteHostClosedError: // 远程主机关闭连接
        QMessageBox::information(this,"提示","远程主机关闭连接",QMessageBox::Yes);
    break;

    case QAbstractSocket::HostNotFoundError: // 找不到主机地址
        QMessageBox::information(this,"提示","找不到主机地址",QMessageBox::Yes);
    break;

    case QAbstractSocket::ConnectionRefusedError: // 连接被对方拒绝（或者超时）
        QMessageBox::information(this,"提示","连接被对方拒绝（或者超时）",QMessageBox::Yes);
    break;

    default:
        QMessageBox::information(this,"提示",tr("致命错误为：").arg(tcpsocket->errorString()),QMessageBox::Yes);

    }

    ui->pushButton_Request->setEnabled(true);
}

void Dialog::opeexec()
{
    ui->lineEdit_ServerName->setEnabled(false);
    tcpsocket->abort(); // 取消已有的连接
    tcpsocket->connectToHost(ui->lineEdit_ServerName->text(),ui->lineEdit_ServerPort->text().toInt());


}
