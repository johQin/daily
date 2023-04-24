#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    mgr=new QNetworkAccessManager(this);

    connect(mgr,SIGNAL(finished(QNetworkReply*)),this,SLOT(replayFinishedFunc(QNetworkReply*)));





}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::replayFinishedFunc(QNetworkReply *reply) // 响应
{
    QString strall=reply->readAll(); // 读取数据
    ui->textBrowser->setText(strall); // 显示数据
    reply->deleteLater();

}

void MainWindow::on_pushButton_GetData_clicked()
{

    ui->label_disp->setText("数据正在下载中，请耐心等待......");
    mgr->get(QNetworkRequest(QUrl("http://www.baidu.com")));
}
