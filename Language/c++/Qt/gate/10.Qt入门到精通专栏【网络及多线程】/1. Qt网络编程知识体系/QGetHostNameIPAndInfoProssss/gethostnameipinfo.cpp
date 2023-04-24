#include "gethostnameipinfo.h"
#include "ui_gethostnameipinfo.h"

GetHostNameIPInfo::GetHostNameIPInfo(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::GetHostNameIPInfo)
{
    ui->setupUi(this);
}

GetHostNameIPInfo::~GetHostNameIPInfo()
{
    delete ui;
}

void GetHostNameIPInfo::GetHostNameAndIpAddress() // 获取主机名称和IP地址
{
    // 获取主机名称
    QString StrLocalHostName=QHostInfo::localHostName();
    ui->lineEdit_hostname->setText(StrLocalHostName);

    // 根据主机名称获取对应的IP地址
    QString StrLocalIpAddress="";
    QHostInfo hostinfo=QHostInfo::fromName(StrLocalHostName);
    QList<QHostAddress> ipaddresslist=hostinfo.addresses();

    if(!ipaddresslist.isEmpty())
    {
        for (int i=0;i<ipaddresslist.size();i++)
        {
            QHostAddress addresshost=ipaddresslist.at(i);
            if(QAbstractSocket::IPv4Protocol==addresshost.protocol())
            {
                StrLocalIpAddress=addresshost.toString();
                break;
            }
        }
    }
    ui->lineEdit_hostip->setText(StrLocalIpAddress);

}

void GetHostNameIPInfo::on_pushButton_GetHostNameIP_clicked()
{
    GetHostNameAndIpAddress();
}

void GetHostNameIPInfo::on_pushButton_GetHostInfo_clicked()
{
    QString strTemp="";

    // 返回主机所找到的所有网络接口的列表
    QList<QNetworkInterface> netlist=QNetworkInterface::allInterfaces();

    for(int i=0;i<netlist.size();i++)
    {
        QNetworkInterface interfaces=netlist.at(i);
        strTemp=strTemp+"设备名称："+interfaces.name()+"\n"; // 获取设备名称
        strTemp=strTemp+"硬件地址："+interfaces.hardwareAddress()+"\n"; // 获取硬件地址

        QList<QNetworkAddressEntry> entrylist=interfaces.addressEntries(); // 遍历每一个IP地址对应信息
        for (int k=0;k<entrylist.count();k++)
        {
            QNetworkAddressEntry etry=entrylist.at(k);

            strTemp=strTemp+"IP地址："+etry.ip().toString()+"\n";
            strTemp=strTemp+"子网掩码："+etry.netmask().toString()+"\n";
            strTemp=strTemp+"广播地址："+etry.broadcast().toString()+"\n";
        }

    }

    QMessageBox::information(this,"主机所有信息",strTemp,QMessageBox::Yes);

}
