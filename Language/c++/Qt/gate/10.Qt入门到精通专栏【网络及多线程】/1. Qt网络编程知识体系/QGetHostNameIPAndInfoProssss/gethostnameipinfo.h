#ifndef GETHOSTNAMEIPINFO_H
#define GETHOSTNAMEIPINFO_H

#include <QDialog>

#include <QHostInfo>
#include <QNetworkInterface>

#include <QMessageBox>

QT_BEGIN_NAMESPACE
namespace Ui { class GetHostNameIPInfo; }
QT_END_NAMESPACE

class GetHostNameIPInfo : public QDialog
{
    Q_OBJECT

public:
    GetHostNameIPInfo(QWidget *parent = nullptr);
    ~GetHostNameIPInfo();

    void GetHostNameAndIpAddress(); // 获取主机名称和IP地址


private slots:
    void on_pushButton_GetHostNameIP_clicked();

    void on_pushButton_GetHostInfo_clicked();

private:
    Ui::GetHostNameIPInfo *ui;
};
#endif // GETHOSTNAMEIPINFO_H
