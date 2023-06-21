//
// Created by buntu on 2023/6/21.
//
#include<iostream>
#include "Rain.h"
#include "DataBS.h"
using namespace std;

void Rain::rainHello(){
    cout<<"rainHello"<<endl;
}
Rain::Rain(){
    DataBS db = DataBS();
//    db.dbHello();
};
//class Rain {
//private:
//    float rainfall = 0.0;
//public:
//    Rain(){
//        MYSQL * conn;
//        MYSQL *MySQLConRet = NULL;
//        conn = mysql_init(NULL);
//        MySQLConRet = mysql_real_connect(conn, "192.168.100.138", "root", "axxt1234", "gasSensor", 3306, NULL, 0);
//        if(NULL == MySQLConRet)
//        {
//            printf("connect is fail.please check......\n");
//        }
//        printf("connect is success.please check......\n");
//    };
//    ~Rain() {
//        cout << "析构函数 rainfall="<< rainfall << endl;
//    };
//public:
//    void hello(){
//        cout<<"hello"<<endl;
//    }
//};