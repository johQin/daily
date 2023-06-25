//
// Created by buntu on 2023/6/21.
//
#include "ReadConfigInfo.h"
//#include "tinyxml.h"
#include<iostream>
#include "Markup.h"
#include<string>
#include<map>
using namespace std;
ReadConfigInfo::ReadConfigInfo(){
    cout<<"ReadConfigInfo constructor"<<endl;
}
ReadConfigInfo::~ReadConfigInfo(){
    cout<<"ReadConfigInfo destructor"<<endl;
}
map<string, string> ReadConfigInfo::getFuncInfo(){
    map<string, string> info;
    string filename = "config.xml";
    CMarkup xml;
    if(!xml.Load("./config.xml")){
        return info;
    }
//    https://blog.51cto.com/u_15045304/5963694
    string stra[] = {"IP","port","userName","pwd","dbName"};
    int len = sizeof(stra) / sizeof(stra[0]);
    for(int i =0 ;i<len;i++){
        xml.FindElem(stra[i]);
    }

    cout<<"ReadConfigInfo getFuncParams"<<endl;

    return info;
}