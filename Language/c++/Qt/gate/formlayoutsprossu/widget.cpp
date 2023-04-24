#include "widget.h"

#include <QFormLayout>
#include <QLineEdit>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{

    setFixedSize(250,200);

    // 创建表单布局指针
    QFormLayout *qLayout=new QFormLayout(this);

    QLineEdit *le1=new QLineEdit(); // 输入学号
    QLineEdit *le2=new QLineEdit(); // 输入姓名
    QLineEdit *le3=new QLineEdit(); // 输入学校

    qLayout->addRow("学号",le1);
    qLayout->addRow("姓名",le2);
    qLayout->addRow("学校",le3);
    qLayout->setSpacing(8);

    // WrapAllRows将标签显示在单行编辑框上面
    // qLayout->setRowWrapPolicy(QFormLayout::WrapAllRows);

    // 当标签和单选编辑框,将标签显示在同一行。
    qLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);


    qLayout->setLabelAlignment(Qt::AlignLeft); // 设置标签对齐方式

    setWindowTitle("表单布局测试案例");

}

Widget::~Widget()
{
}

