#include <QCoreApplication>//QT 提供一个事件循环
#include<QDebug> //输出流
//#include<QString>
#include<QDateTime>
#include<iostream>
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // 1. QString
    QString str1 = "I ";
    str1 += "love you";             // 等价于 str1 = str1 + "love you"
    qDebug()<< str1;                //打印出的字符有双引号
    qDebug()<<qPrintable(str1);     //字符串无双引号

    // 2. QString::append(), 等价于+=
    //相关函数：prepend/push_front前向追加，append/push_back
    QString str2 = "I love you";
    str1.append(", me too");        //I love you, me too

    // 3. QString::sprintf()，组合字符串
    //  和c++中的string.h中的sprintf等价
    QString str3 = "I love you";
    str3.sprintf("%s haha %s", "happy 4.5","cheat you"); //happy 4.5 haha cheat you

    // 4. QString::arg()
    QString str4 = QString("%1 is %2 of %3").arg(2000).arg("number").arg("real");
    //   2000 is number of real


    // 5. QString::startsWith()
    QString str5 = "BeiJing welcome to you";

    str5.startsWith("BeiJing",Qt::CaseSensitive);       //true
    str5.endsWith("yo",Qt::CaseSensitive);              //false
    str5.contains("welcome",Qt::CaseSensitive);         //true

    //6. QString::toInt()
    QString str6 = "17";
    bool toIsSuccess = false;
    int hex= str6.toInt(&toIsSuccess,16);
    // 这里的16表示的是原数17的进制是16进制的，0x17 => 23
    // toIsSuccess = true,hex = 23,

    // 7. QString::compare()
    int a1 = QString::compare("abcd","abcd",Qt::CaseSensitive);//a与a的差值，为0
    int b1 = QString::compare("abcd","ABCD",Qt::CaseSensitive);//a（97）与A（65）的差值，32
    int c1 = QString::compare("abcd","c",Qt::CaseInsensitive);//a与c的差值，为-2
    //cout<<"a1= "<<a1<<" b1= "<<b1<<" c1= "<<c1<<endl;// a1= 0 b1= 32 c1= -2

    // 8. QString::toUtf8()
    QString str = "ABC abc";
    QByteArray bytes = str.toUtf8();
    for (int i =0; i<str.size();++i){
        qDebug()<<bytes.at(i);
    }

    // 9. QDateTime
    QDateTime dt;
    QString dtStr = dt.currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    qDebug()<<dtStr;

    return a.exec();
}
