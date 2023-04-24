#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>


#include <QMessageBox>
#include "tcpserverp.h"
class tcpserverp;





QT_BEGIN_NAMESPACE
namespace Ui { class Dialog; }
QT_END_NAMESPACE

class Dialog : public QDialog
{
    Q_OBJECT

public:
    Dialog(QWidget *parent = nullptr);
    ~Dialog();

private:
    Ui::Dialog *ui;



private:
    int icount; // 统计客户端访问次数
    tcpserverp *tcpserver; // 创建服务对象

public slots:
    void slotsdispFunc();


















};
#endif // DIALOG_H
