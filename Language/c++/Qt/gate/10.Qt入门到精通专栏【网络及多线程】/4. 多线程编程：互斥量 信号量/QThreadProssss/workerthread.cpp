#include "workerthread.h"

#include <QtDebug>

WorkerThread::WorkerThread()
{

}

void WorkerThread::run()
{
    while(true)
    {
        for(int i=1;i<=5;i++)
            qDebug()<<i<<i<<i<<i<<i;
    }
}
