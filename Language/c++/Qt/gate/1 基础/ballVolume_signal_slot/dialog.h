#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>

// 引入标签，命令按钮等对应头文件
#include<qlabel.h>
#include<qpushbutton.h>
#include<qlineedit.h>

class Dialog : public QDialog
{
    Q_OBJECT

public:
    Dialog(QWidget *parent = nullptr);
    ~Dialog();

private:
    QLabel *lab1,*lab2;
    QLineEdit *lEdit;
    QPushButton *pbt;

private slots:
    void CalculateVolume();//计算圆球的体积
};
#endif // DIALOG_H
