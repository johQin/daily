#include "dialog.h"
#include<QGridLayout>
const static double PI = 3.1415;
Dialog::Dialog(QWidget *parent)
    : QDialog(parent)
{
    lab1 = new QLabel(this);
    lab1->setText(tr("请输入圆球半径"));

    lab2 = new QLabel(this);

    lEdit = new QLineEdit(this);
    pbt =new QPushButton(this);
    pbt->setText(tr("计算圆球的体积"));
    QGridLayout *mLay = new QGridLayout(this);
    mLay->addWidget(lab1,0,0);
    mLay->addWidget(lEdit,0,1);
    mLay->addWidget(lab2,1,0);
    mLay->addWidget(pbt,1,1);

    connect(lEdit,SIGNAL(textChanged(QString)), this,SLOT(CalculateVolume()));
}

Dialog::~Dialog()
{
    delete lab1;
    delete lab2;
    delete lEdit;
    delete pbt;
}

void Dialog::CalculateVolume()
{
    bool isLoop;
    QString tempStr;
    QString valueStr = lEdit->text();
    int valueInt = valueStr.toInt(&isLoop);
    double dv = 4 *PI* valueInt * valueInt * valueInt /3;
    lab2->setText(tempStr.setNum(dv));
}
