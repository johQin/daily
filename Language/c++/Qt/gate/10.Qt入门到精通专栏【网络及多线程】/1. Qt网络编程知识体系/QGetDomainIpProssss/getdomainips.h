#ifndef GETDOMAINIPS_H
#define GETDOMAINIPS_H

#include <QDialog>

#include <QHostInfo>

QT_BEGIN_NAMESPACE
namespace Ui { class GetDomainIps; }
QT_END_NAMESPACE

class GetDomainIps : public QDialog
{
    Q_OBJECT

public:
    GetDomainIps(QWidget *parent = nullptr);
    ~GetDomainIps();

    // 获取IP地址类型
    QString ProtocolTypeName(QAbstractSocket::NetworkLayerProtocol pro);

private slots:
    // 获取IP地址列表
    void LookupHostInfoFunc(const QHostInfo &host);

private slots:
    void on_pushButton_ClearData_clicked();

    void on_pushButton_GetDomainIP_clicked();

private:
    Ui::GetDomainIps *ui;
};
#endif // GETDOMAINIPS_H
