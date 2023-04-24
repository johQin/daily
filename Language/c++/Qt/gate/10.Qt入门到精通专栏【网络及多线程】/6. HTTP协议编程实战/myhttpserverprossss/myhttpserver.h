#ifndef MYHTTPSERVER_H
#define MYHTTPSERVER_H

#include <QObject>

#include <QTcpServer>
#include <QTcpSocket>


class myhttpserver : public QObject
{
    Q_OBJECT
public:
    explicit myhttpserver(QObject *parent = nullptr);
    ~myhttpserver();


public:
    QTcpSocket *socket;

private:
    QTcpServer *ser;

public slots:
    void connection(); // 连接




};

#endif // MYHTTPSERVER_H
