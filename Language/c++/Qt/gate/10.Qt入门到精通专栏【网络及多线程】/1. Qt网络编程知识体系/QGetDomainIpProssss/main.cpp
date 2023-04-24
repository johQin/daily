#include "getdomainips.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GetDomainIps w;
    w.show();
    return a.exec();
}
