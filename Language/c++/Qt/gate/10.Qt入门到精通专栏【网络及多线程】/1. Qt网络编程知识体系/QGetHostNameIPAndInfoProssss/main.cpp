#include "gethostnameipinfo.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GetHostNameIPInfo w;
    w.show();
    return a.exec();
}
