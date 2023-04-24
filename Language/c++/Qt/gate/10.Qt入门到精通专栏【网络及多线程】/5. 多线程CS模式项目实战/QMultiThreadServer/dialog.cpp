#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::Dialog)
{
    ui->setupUi(this);

    icount=0;

    tcpserver =new tcpserverp(this);
    if(!tcpserver->listen())
    {
        QMessageBox::critical(this,tr("提示"),
                              tr("多线程服务器无法启动，请重新检查").arg(tcpserver->errorString()));
        close();
        return;
    }

    ui->lineEdit_ServerPort->setText(tr("%1").arg(tcpserver->serverPort()));

    connect(ui->pushButton_Exit,SIGNAL(clicked()),this,SLOT(close()));



}

Dialog::~Dialog()
{
    delete ui;
}


void Dialog::slotsdispFunc()
{
    ui->lineEdit_Count->setText(tr("客户端请求%1次").arg(++icount));

}
