#include "getdomainips.h"
#include "ui_getdomainips.h"

GetDomainIps::GetDomainIps(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::GetDomainIps)
{
    ui->setupUi(this);

    ui->lineEdit_InputUrl->setText("www.126.com");
}

GetDomainIps::~GetDomainIps()
{
    delete ui;
}

// 网络层协议
QString GetDomainIps::ProtocolTypeName(QAbstractSocket::NetworkLayerProtocol protocoltype)
{
    switch (protocoltype)
    {
    case QAbstractSocket::IPv4Protocol:
        return "IPv4 Protocol";
    case QAbstractSocket::IPv6Protocol:
        return "IPv6 Protocol";
    case QAbstractSocket::AnyIPProtocol:
        return "Any IP Protocol";
    default:
        return "Unknown Network Layer Protocol";
    }
}

void GetDomainIps::on_pushButton_ClearData_clicked()
{
    ui->plainTextEdit_DomainIP->clear();
}

// 获取IP地址列表
void GetDomainIps::LookupHostInfoFunc(const QHostInfo &host)
{
    QList<QHostAddress> addresslist=host.addresses();
    for(int i=0;i<addresslist.count();i++)
    {
        QHostAddress host=addresslist.at(i);

        ui->plainTextEdit_DomainIP->appendPlainText("协议类型："+ProtocolTypeName(host.protocol()));
        ui->plainTextEdit_DomainIP->appendPlainText("本地IP地址："+host.toString());
        ui->plainTextEdit_DomainIP->appendPlainText("");
    }
}





void GetDomainIps::on_pushButton_GetDomainIP_clicked()
{
    // 主机名称
    QString strhostname=ui->lineEdit_InputUrl->text();
    ui->plainTextEdit_DomainIP->appendPlainText("你所查询主机信息："+strhostname);
    QHostInfo::lookupHost(strhostname,this,SLOT(LookupHostInfoFunc(QHostInfo)));
}






