#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>


#include <QTcpSocket>
#include <QAbstractSocket>
#include <QDataStream>
#include <QMessageBox>


QT_BEGIN_NAMESPACE
namespace Ui { class Dialog; }
QT_END_NAMESPACE

class Dialog : public QDialog
{
    Q_OBJECT

public:
    Dialog(QWidget *parent = nullptr);
    ~Dialog();

private slots:
    void on_pushButton_Request_clicked();

private:
    Ui::Dialog *ui;

    QTcpSocket *tcpsocket;

public slots:
    void dispError(QAbstractSocket::SocketError socketerror);
    void opeexec();






};
#endif // DIALOG_H
