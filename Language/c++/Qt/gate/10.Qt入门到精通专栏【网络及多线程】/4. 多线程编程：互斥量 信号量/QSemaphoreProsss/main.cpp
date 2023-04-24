#include <QCoreApplication>

#include <QThread>
#include <QSemaphore>
#include <QTime>
#include <iostream>


const int datasize=100;
const int buffersize=1;

QSemaphore freesapce(buffersize);
QSemaphore usedspace(0);

class producer:public QThread // 生产者
{
protected:
    void run()
    {
        qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
        qsrand(NULL);

        for(int i=0;i<datasize;i++)
        {
            freesapce.acquire(); // 获取资源
            std::cerr<<i<<":producer-->";
            usedspace.release(); // 释放资源
        }
    }
};

class consumers : public QThread
{
protected:
    void run()
    {
        for(int i=0;i<datasize;i++)
        {
            usedspace.acquire(); // 获取资源
            std::cerr<<i<<":consumers\n";
            freesapce.release(); // 释放资源
        }
    }
};


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    producer p;
    consumers c;
    p.start();
    c.start();
    p.wait();
    c.wait();


    return a.exec();
}
