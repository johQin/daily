//
// Created by buntu on 2023/6/21.
//
#include "DataBS.h"
#include<iostream>
#include <mysql.h>
#include "ReadConfigInfo.h"
using namespace std;
DataBS::DataBS(){
    cout<<"DataBS constructor"<<endl;
    ReadConfigInfo rc = ReadConfigInfo();
    rc.getFuncInfo();
//    MYSQL * conn;
//    MYSQL *MySQLConRet = NULL;
//    conn = mysql_init(NULL);
//    MySQLConRet = mysql_real_connect(conn, "127.0.0.1", "root", "root", "ps", 3306, NULL, 0);
//    if(NULL == MySQLConRet)
//    {
//        printf("connect is fail.please check......\n");
//    }
//    printf("connect is success.please check......\n");
}
DataBS::~DataBS(){
    cout<<"DataBS destructor"<<endl;
}
void DataBS::dbHello(){
    cout<<"dbHello"<<endl;
}