#include "widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    button1=new QPushButton(this);
    button1->setText("第一区：顶部菜单栏选项");
    button1->setFixedHeight(40); // 设置固定大小高度
    button1->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

    button2=new QPushButton(this);
    button2->setText("第二区：侧边栏选项");
    button2->setFixedWidth(100); // 设置固定大小宽度
    button2->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

    button3=new QPushButton(this);
    button3->setText("第三区：底部选项");
    button3->setFixedHeight(40);
    button3->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

    button4=new QPushButton(this);
    button4->setText("第四区：子窗体选项");
    button3->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);

    pGrid_layouts=new QGridLayout();

    // 通过此函数设置左侧 顶部 右侧 底部边距，主要方便布局周围进行使用
    pGrid_layouts->setContentsMargins(0,0,0,0);


    // pGrid_layouts->setMargin(30);
    // pGrid_layouts->setSpacing(40);

    pGrid_layouts->setSpacing(0);


    // 显示位置
    // addWidget(参数1，参数2，参数3，参数4，参数5，参数6)
    /*
    1:我要插入的子布局对象
    2:插入的开始行
    3:插入的开始列
    4:占用的行数
    5:占用的列数
    6:指定对齐方式
    */
    pGrid_layouts->addWidget(button1,0,1);
    pGrid_layouts->addWidget(button2,0,0,3,1);
    pGrid_layouts->addWidget(button3,2,1);
    pGrid_layouts->addWidget(button4,1,1);

    setLayout(pGrid_layouts);

}

Widget::~Widget()
{
}

