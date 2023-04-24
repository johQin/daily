#include "udpcomm.h"
#include "ui_udpcomm.h"

#include <QMessageBox>

udpComm::udpComm(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::udpComm)
{
    ui->setupUi(this);

    QString strip=GetLocalIpAddress(); // 调用此函数返回对应本机IP地址
    // QMessageBox::information(this,"结果",strip,QMessageBox::Yes);

    ui->comboBoxtargetip->addItem(strip); // 将IP地址显示到comboBox控件

    udpsocket=new QUdpSocket(this);

    connect(udpsocket,SIGNAL(readyRead()),this,SLOT(SocketReadyReadData()));


}

udpComm::~udpComm()
{
    delete ui;
}

// 启动服务
void udpComm::on_pushButton_start_clicked()
{
    quint16 port=ui->spinBoxbindport->value(); // 本机UDP端口

    if(udpsocket->bind(port))
    {
        ui->plainTextEditdispmsg->appendPlainText("**********绑定成功**********");
        ui->plainTextEditdispmsg->appendPlainText("$$$$$$$$$$绑定端口$$$$$$$$$$："+
                                                  QString::number(udpsocket->localPort()));
        ui->pushButton_start->setEnabled(false);
        ui->pushButton_stop->setEnabled(true);

    }
    else
    {
        ui->plainTextEditdispmsg->appendPlainText("**********绑定失败**********");
    }

}

void udpComm::on_pushButton_stop_clicked()
{
    udpsocket->abort();
    ui->pushButton_start->setEnabled(true);
    ui->pushButton_stop->setEnabled(false);
    ui->plainTextEditdispmsg->appendPlainText("**********已经停止服务**********");

}

void udpComm::on_pushButton_sendmsg_clicked()
{
    QString targetIpAddress=ui->comboBoxtargetip->currentText(); // 获取目标IP地址
    QHostAddress targetaddress(targetIpAddress);

    quint16 targetport=ui->spinBoxtargetport->value(); // 获取端口

    QString strmsg=ui->lineEditmsg->text(); // 获取发送消息内容

    QByteArray str=strmsg.toUtf8();

    udpsocket->writeDatagram(str,targetaddress,targetport); // 发送数据报信息

    ui->plainTextEditdispmsg->appendPlainText("[out]："+str);
    ui->lineEditmsg->clear(); // 清除编辑框控件内容
    ui->lineEditmsg->setFocus(); // 将光标焦点定位到编辑框控件

}

void udpComm::on_pushButton_broadcastmsg_clicked()
{
    quint16 targetport=ui->spinBoxtargetport->value(); // 获取端口
    QString strmsg=ui->lineEditmsg->text(); // 获取发送消息内容
    QByteArray str=strmsg.toUtf8();

    udpsocket->writeDatagram(str,QHostAddress::Broadcast,targetport); // 发送数据报信息

    ui->plainTextEditdispmsg->appendPlainText("[broadcast]："+str);
    ui->lineEditmsg->clear(); // 清除编辑框控件内容
    ui->lineEditmsg->setFocus(); // 将光标焦点定位到编辑框控件


}



QString udpComm::GetLocalIpAddress() // 获取本机IP地址
{
    // 根据主机名称，获取IP地址
    QString strHostName=QHostInfo::localHostName();
    QHostInfo hostinfo=QHostInfo::fromName(strHostName); //  通过主机名称获取IP地址

    QString strLocalIp="";

    QList<QHostAddress> addresslist=hostinfo.addresses(); // IP地址列表

    if(!addresslist.isEmpty())
    {
        for(int i=0;i<addresslist.count();i++)
        {
            QHostAddress hostaddr=addresslist.at(i);

            if(QAbstractSocket::IPv4Protocol==hostaddr.protocol())
            {
                strLocalIp=hostaddr.toString();
                break;
            }
        }
    }

    return strLocalIp;
}

void  udpComm::SocketReadyReadData() // 读取socket传输数据信息
{
    // 读取接收到的数据报信息
    // 用此函数返回true至少有一个数据报需要读取
    while(udpsocket->hasPendingDatagrams())
    {
        QByteArray datagrams;

        datagrams.resize(udpsocket->pendingDatagramSize());

        QHostAddress paddress;
        quint16 pport;

        // 通过readDatagram()此函数读取数据报，
        udpsocket->readDatagram(datagrams.data(),datagrams.size(),&paddress,&pport);

        QString strs=datagrams.data();
        QString peer="[From:"+paddress.toString()+":"+QString::number(pport)+"]:";

        ui->plainTextEditdispmsg->appendPlainText(peer+strs);
    }

}

















