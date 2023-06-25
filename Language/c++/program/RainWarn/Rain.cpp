//
// Created by buntu on 2023/6/21.
//
#include<iostream>
#include "Rain.h"
#include "DataBS.h"
#include "ReadConfigInfo.h"
#include <map>
#include<list>
#include<vector>
using namespace std;

//void Rain::rainHello(){
//    cout<<"rainHello"<<endl;
//}
Rain::Rain(){

    ReadConfigInfo rc = ReadConfigInfo();
    funInfoMap= rc.getFuncInfo("RainWarn");
};
void Rain::rainWarn(){
    DataBS db = DataBS();
    string sqlStr = "select * from " + funInfoMap["tableName"] + " where cs_data_time between " + funInfoMap["startTime"] + \
            " and " + funInfoMap["endTime"] + " order by cs_data_time desc";
    list<vector<string>> res = db.query(sqlStr);

}
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