//
// Created by buntu on 2023/6/21.
//
#include "DataBS.h"
#include<iostream>
#include <string>
#include <mysql.h>
#include "ReadConfigInfo.h"         //这里不能写utils/xml/include/ReadConfigInfo.h
#include<map>
#include<list>
#include<vector>
#include<stdlib.h>
using namespace std;
DataBS::DataBS(){
    ReadConfigInfo rc = ReadConfigInfo();
    rc.getHostInfo(dbInfoMap);
}
DataBS::~DataBS(){}
int DataBS::query(string sqlStr, callback call_fun){
    MYSQL * conn;
    MYSQL *MySQLConRet = NULL;
    conn = mysql_init(NULL);
    MySQLConRet = mysql_real_connect(conn, dbInfoMap["IP"].c_str(), dbInfoMap["userName"].c_str(), dbInfoMap["pwd"].c_str(), dbInfoMap["dbName"].c_str(), stoi(dbInfoMap["port"]), NULL, 0);
    list<vector<string>> resList;
    if(NULL == MySQLConRet)
    {
        return -1;
    }
    printf("connect is success\n");
    mysql_query(conn, sqlStr.c_str());
    MYSQL_RES *result = mysql_store_result(conn);
    int num_fields = mysql_num_fields(result);
    MYSQL_ROW row;
    while ((row = mysql_fetch_row(result))){
        vector<string> tmp;
        for(int i = 0; i < num_fields; i++) {
            tmp.reserve(num_fields);
            if (row[i]) {
                tmp.push_back(row[i]);
            } else {
                tmp.push_back("");
            }
        }
        resList.push_back(tmp);
    }
    call_fun(resList);
    mysql_free_result(result);
    mysql_close(conn);
    return 1;
}