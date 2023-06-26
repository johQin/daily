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
#include "Utils.h"
using namespace std;

//void Rain::rainHello(){
//    cout<<"rainHello"<<endl;
//}
Rain::Rain(){

    ReadConfigInfo rc = ReadConfigInfo();
    funInfoMap= rc.getFuncInfo("RainWarn");
};
int Rain::rainDataHandleCallback(list<vector<string>> list){
    cout<<"rainDataHandleCallback"<<endl;
    return 1;
}
void Rain::rainWarn(){
    DataBS db = DataBS();
    char buf[200]={'\0'};
    sprintf(buf,"select * from %s where cs_data_time between %ld and %ld order by cs_data_time desc limit 1", funInfoMap["tableName"],TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["startTime"]),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["endTime"]));
    string sqlStr = string(buf);
    db.query("select * from rainfall",rainDataHandleCallback);

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