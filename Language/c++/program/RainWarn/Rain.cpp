//
// Created by buntu on 2023/6/21.
//
#include<iostream>
#include "Rain.h"
#include "DataBS.h"
#include "ReadConfigInfo.h"
#include <map>
#include<list>
#include <stdlib.h>
#include<vector>
#include "Utils.h"
#include <functional>
using namespace std;

Rain::Rain(){

    ReadConfigInfo rc = ReadConfigInfo();
    rc.getFuncInfo("RainWarn", funInfoMap);
};
int Rain::rainDataHandleCallback(list<vector<string>> &list){
    if(list.empty()) return -1;
    vector<string> rainfallRecord =list.front();
    double rf = atof(rainfallRecord[1].c_str());
    if(rf > atof(funInfoMap["rainThreshold"].c_str())){

    }
    return 1;
}
void Rain::rainWarn(){
    DataBS db = DataBS();
    char buf[200]={'\0'};
    sprintf(buf,"select * from %s where cs_data_time between %ld and %ld order by cs_data_time desc limit 1", funInfoMap["tableName"].c_str(),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["startTime"]),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["endTime"]));
    string sqlStr = string(buf);
    db.query(sqlStr,bind(&Rain::rainDataHandleCallback, this, std::placeholders::_1));

}
