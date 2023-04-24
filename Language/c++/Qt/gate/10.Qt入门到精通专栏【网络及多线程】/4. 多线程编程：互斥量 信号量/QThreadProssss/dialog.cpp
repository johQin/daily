#include "dialog.h"

Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("QThread类线程操作");

    startbutton=new QPushButton("开 始");
    stopbutton=new QPushButton("停 止");
    exitbutton=new QPushButton("退 出");

    QHBoxLayout *hlayout=new QHBoxLayout(this);
    hlayout->addWidget(startbutton);
    hlayout->addWidget(stopbutton);
    hlayout->addWidget(exitbutton);

    connect(startbutton,SIGNAL(clicked()),this,SLOT(onSlotStart()));
    connect(stopbutton,SIGNAL(clicked()),this,SLOT(onSlotStop()));
    connect(exitbutton,SIGNAL(clicked()),this,SLOT(close()));

}

Dialog::~Dialog()
{
}


void Dialog::onSlotStart()
{
    for(int i=0;i<MAXSIZE;i++)
    {
        workerthread[i]=new WorkerThread();
    }
    for(int i=0;i<MAXSIZE;i++)
    {
        workerthread[i]->start();
    }

    startbutton->setEnabled(false);
    stopbutton->setEnabled(true);

}

void Dialog::onSlotStop()
{
    for(int i=0;i<MAXSIZE;i++)
    {
        workerthread[i]->terminate();
        workerthread[i]->wait();
    }
    startbutton->setEnabled(true);
    stopbutton->setEnabled(false);

}
