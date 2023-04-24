#include "udpcomm.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    udpComm w;
    w.show();
    return a.exec();
}
