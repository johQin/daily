#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    tcpclient=new QTcpSocket(this);

    QString strip=getlocalip();

    ui->comboBoxIp->addItem(strip);


    connect(tcpclient,SIGNAL(connected()),this,SLOT(connectfunc()));
    connect(tcpclient,SIGNAL(disconnected()),this,SLOT(disconnectfunc()));
    connect(tcpclient,SIGNAL(readyRead()),this,SLOT(socketreaddata()));





}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_Connect_clicked()
{
    QString addr=ui->comboBoxIp->currentText();
    quint16 port=ui->spinBoxPort->value();
    tcpclient->connectToHost(addr,port);
}

void MainWindow::on_pushButton_Send_clicked()
{
    QString strmsg=ui->lineEdit_InputMsg->text();
    ui->plainTextEdit_DispMsg->appendPlainText("[out]:"+strmsg);
    ui->lineEdit_InputMsg->clear();

    QByteArray str=strmsg.toUtf8();
    str.append('\n');
    tcpclient->write(str);

}


void MainWindow::on_pushButton_Disconnect_clicked()
{
    if(tcpclient->state()==QAbstractSocket::ConnectedState)
        tcpclient->disconnectFromHost();
}




QString MainWindow::getlocalip() // 获取本机IP地址
{
    QString hostname=QHostInfo::localHostName();
    QHostInfo hostinfo=QHostInfo::fromName(hostname);

    QString localip="";

    QList<QHostAddress> addlist=hostinfo.addresses();
    if(!addlist.isEmpty())
    {
        for (int i=0;i<addlist.count();i++)
        {
            QHostAddress ahost=addlist.at(i);
            if(QAbstractSocket::IPv4Protocol==ahost.protocol())
            {
                localip=ahost.toString();
                break;
            }
        }
    }

    return localip;
}


void MainWindow::closeEvent(QCloseEvent *event)
{
    if(tcpclient->state()==QAbstractSocket::ConnectedState)
    {
        tcpclient->disconnectFromHost();
    }
    event->accept();

}


void MainWindow::connectfunc()
{
    ui->plainTextEdit_DispMsg->appendPlainText("**********已经连接到服务器端**********");
    ui->plainTextEdit_DispMsg->appendPlainText("**********peer address:"+
                                               tcpclient->peerAddress().toString());
    ui->plainTextEdit_DispMsg->appendPlainText("**********peer port:"+
                                               QString::number(tcpclient->peerPort()));

    ui->pushButton_Connect->setEnabled(false);
    ui->pushButton_Disconnect->setEnabled(true);

}
void MainWindow::disconnectfunc()
{
    ui->plainTextEdit_DispMsg->appendPlainText("**********已断开与服务器端的连接**********");

    ui->pushButton_Connect->setEnabled(true);
    ui->pushButton_Disconnect->setEnabled(false);

}
void MainWindow::socketreaddata()
{
    while(tcpclient->canReadLine())
        ui->plainTextEdit_DispMsg->appendPlainText("[in]:"+tcpclient->readLine());

}
