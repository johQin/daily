//
// Created by buntu on 2023/6/21.
//
#include "DataBS.h"
#include<iostream>
#include <string>
#include <mysql.h>
#include "ReadConfigInfo.h"
#include<map>
#include<list>
#include<vector>
#include<stdlib.h>
using namespace std;
DataBS::DataBS(){
    cout<<"DataBS constructor"<<endl;
    ReadConfigInfo rc = ReadConfigInfo();
    rc.getHostInfo(dbInfoMap);
}
DataBS::~DataBS(){
    cout<<"DataBS destructor"<<endl;
}
int DataBS::query(string sqlStr, callback call_fun){
    cout<<"dbHello"<<endl;
    cout<<dbInfoMap["IP"]<<endl;
    MYSQL * conn;
    MYSQL *MySQLConRet = NULL;
    conn = mysql_init(NULL);
    MySQLConRet = mysql_real_connect(conn, dbInfoMap["IP"].c_str(), dbInfoMap["userName"].c_str(), dbInfoMap["pwd"].c_str(), dbInfoMap["dbName"].c_str(), stoi(dbInfoMap["port"]), NULL, 0);
    list<vector<string>> resList;
    if(NULL == MySQLConRet)
    {
        printf("connect is fail.please check......\n");
        return -1;
    }
    printf("connect is success.please check......\n");
    mysql_query(conn, "SELECT * FROM rainfall");
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