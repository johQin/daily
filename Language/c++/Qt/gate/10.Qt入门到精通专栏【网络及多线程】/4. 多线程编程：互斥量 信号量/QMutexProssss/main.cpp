#include <QCoreApplication>

#include <iostream>
#include <QMutex>
#include <QThread>
#include <QObject>

// 售票类
class ticketsllers:public QObject
{
public:
    ticketsllers();
    ~ticketsllers();

public slots:
    void salefunc(); // 售票

public:
    int *tickets; // 票
    QMutex *metx; // 互斥量
    std::string sellersname; // 售票员姓名
};
ticketsllers::ticketsllers()
{
    metx=NULL;
    tickets=0;
}

ticketsllers::~ticketsllers()
{

}

void ticketsllers::salefunc()
{
    while((*tickets)>0)
    {
        metx->lock();
        std::cout<<sellersname<<" : " <<(*tickets)-- << std::endl;

        metx->unlock();
    }
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    int ticket=20;
    QMutex mtx;

    // 创建线程1
    QThread th1;
    ticketsllers seller1;
    // 设置线程
    seller1.tickets=&ticket;
    seller1.metx=&mtx;
    seller1.sellersname="seller kitty";
    // 将对象移动到线程
    seller1.moveToThread(&th1);

    // 创建线程1
    QThread th2;
    ticketsllers seller2;
    // 设置线程
    seller2.tickets=&ticket;
    seller2.metx=&mtx;
    seller2.sellersname="seller sunny";
    // 将对象移动到线程
    seller2.moveToThread(&th2);

    // 创建线程3
    QThread th3;
    ticketsllers seller3;
    // 设置线程
    seller3.tickets=&ticket;
    seller3.metx=&mtx;
    seller3.sellersname="seller andy";
    // 将对象移动到线程
    seller3.moveToThread(&th3);


    QObject::connect(&th1,&QThread::started,&seller1,&ticketsllers::salefunc);
    QObject::connect(&th2,&QThread::started,&seller2,&ticketsllers::salefunc);
    QObject::connect(&th3,&QThread::started,&seller3,&ticketsllers::salefunc);

    th1.start();
    th2.start();
    th3.start();

    return a.exec();
}
