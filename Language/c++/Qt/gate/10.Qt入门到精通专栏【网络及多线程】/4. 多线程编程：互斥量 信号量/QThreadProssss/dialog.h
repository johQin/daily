#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>

#include "workerthread.h"
#include <QPushButton>
#include <QHBoxLayout>

#define MAXSIZE 5

class Dialog : public QDialog
{
    Q_OBJECT

public:
    Dialog(QWidget *parent = nullptr);
    ~Dialog();

private:
    QPushButton *startbutton;
    QPushButton *stopbutton;
    QPushButton *exitbutton;

    WorkerThread *workerthread[MAXSIZE];

public slots:
    void onSlotStart();
    void onSlotStop();




};
#endif // DIALOG_H
