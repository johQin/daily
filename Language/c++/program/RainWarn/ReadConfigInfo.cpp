//
// Created by buntu on 2023/6/21.
//
#include "ReadConfigInfo.h"
#include "tinyxml.h"
#include<iostream>
#include<map>
using namespace std;
ReadConfigInfo::ReadConfigInfo(){
    cout<<"ReadConfigInfo constructor"<<endl;
}
ReadConfigInfo::~ReadConfigInfo(){
    cout<<"ReadConfigInfo destructor"<<endl;
}
map<string, string> ReadConfigInfo::getFuncInfo(){
//    TiXmlDocument doc;
    cout<<"ReadConfigInfo getFuncParams"<<endl;
    map<string, string> info;
    return info;
}