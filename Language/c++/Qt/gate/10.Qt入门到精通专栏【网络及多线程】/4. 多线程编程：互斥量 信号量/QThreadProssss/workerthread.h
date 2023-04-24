#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H

#include <QThread>

class WorkerThread : public QThread
{
    Q_OBJECT
public:
    WorkerThread();

protected:
    void run();

};

#endif // WORKERTHREAD_H
