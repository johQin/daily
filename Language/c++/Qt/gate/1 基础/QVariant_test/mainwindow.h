#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
struct Student{
    int no;
    QString name;
};
Q_DECLARE_METATYPE(Student)
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
};
#endif // MAINWINDOW_H
