//
// Created by buntu on 2023/6/21.
//
#include "ReadConfigInfo.h"
//#include "tinyxml.h"
#include<iostream>
#include "Markup.h"
#include<string>
#include<map>
#include<cstdlib>
#include "SelfENV.h"
#include<stdio.h>
#include "Utils.h"
#include <time.h>
#include <iomanip>
using namespace std;
ReadConfigInfo::ReadConfigInfo(){}
ReadConfigInfo::~ReadConfigInfo(){}

int ReadConfigInfo::getHostInfo(map<string,string> & info){
//    map<string, const char *> info;
    CMarkup xml;
    string configXmlPath = getProjectRootPath() + "/config.xml";

    if(!xml.Load(configXmlPath)){
        return -1;
    }
    xml.ResetPos();
    // 然后再从根节点开始往下找
    xml.FindElem("body");
    xml.IntoElem();
    xml.FindElem("param");

//    https://blog.51cto.com/u_15045304/5963694
    string stra[] = {"IP","port","userName","pwd","dbName"};
    int len = sizeof(stra) / sizeof(stra[0]);
    for(int i =0 ;i<len;i++){
        xml.IntoElem();
        bool flag= xml.FindElem(stra[i]);
        if (!flag){
            continue;
        }
//        const char * ciValue = xml.GetAttrib("value").c_str();
        info[stra[i]] = xml.GetAttrib("value");
//        info.insert(make_pair(stra[i], ciValue));     // 因为info的value为const char *类型，所以如果用这里的make_pair方式会出现value为空字符串，只能通过上面的方式
        xml.OutOfElem();
    }
    return 1;
}

int ReadConfigInfo::getFuncInfo(string funcName,map<string,string> & info){
    CMarkup xml;
    string configXmlPath = getProjectRootPath() + "/config.xml";
    if(!xml.Load(configXmlPath)){
        return -1;
    }
    // 然后再从根节点开始往下找
    xml.FindElem("body");
    xml.IntoElem();
    xml.FindElem("funcModels");
    xml.IntoElem();
    while(xml.FindElem("Item")){
        if(xml.GetAttrib("func") == funcName){
            MCD_STR strName, strAttrib;
            int n = 0;
            // 获取标签的所有属性值
            while ( xml.GetNthAttrib(n++, strName, strAttrib) )
            {
                info.insert(make_pair(strName, strAttrib));
            }
            break;
        }
    }
    return 1;
}
string ReadConfigInfo::getProjectRootPath(){
    char *path = getenv("RESOURCE_DIR");
    if (path != nullptr) {
        return path;
    }

    // not find in ENV
    string project_path = PROJECT_ROOT_PATH;
    if(project_path.c_str()){
        return project_path;
    }

    return "";
}