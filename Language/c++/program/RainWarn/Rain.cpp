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
int Rain::personInMineHandleCallback(std::list<std::vector<std::string>> &list) {
    std::list<std::vector<std::string>>::iterator it = list.begin();
    std::map<string, string> personMap;
    //list是双向链表，它的迭代器是双向迭代器，不支持+2操作，支持++操作
    for (; it != list.end(); it++) {
        personMap[(*it)[0]] = (*it)[1];
        std::cout<<(*it)[1]<<std::endl;
    }
    std::list<string> personInMine;
    for(auto it : personMap){
        if(it.second.compare("1") ){
            personInMine.push_back(it.first);
        }
    }
    if(personInMine.size() > 0){

    }

}
int Rain::rainDataHandleCallback(list<vector<string>> &list){
    if(list.empty()) return -1;
    vector<string> rainfallRecord =list.front();
    double rf = atof(rainfallRecord[0].c_str());
    if(rf > atof(funInfoMap["rainThreshold"].c_str())){
        char buf[200]={'\0'};
        sprintf(buf,"select ps_person_card,ps_enter_flag from %s where cs_data_time between %ld and %ld order by cs_data_time asc", funInfoMap["personTableName"].c_str(),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["startTime"]),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["endTime"]));
        string sqlStr = string(buf);
        DataBS db = DataBS();
        db.query(sqlStr,bind(&Rain::personInMineHandleCallback, this, std::placeholders::_1));
    }
    return 1;
}
void Rain::rainWarn(){
    DataBS db = DataBS();
    char buf[200]={'\0'};
    sprintf(buf,"select rainfall from %s where cs_data_time between %ld and %ld order by cs_data_time desc limit 1", funInfoMap["tableName"].c_str(),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["startTime"]),TimeTransfer::convertTimeStr2TimeStamp(funInfoMap["endTime"]));
    string sqlStr = string(buf);
    db.query(sqlStr,bind(&Rain::rainDataHandleCallback, this, std::placeholders::_1));

}
