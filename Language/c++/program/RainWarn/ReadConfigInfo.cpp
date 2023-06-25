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
using namespace std;
ReadConfigInfo::ReadConfigInfo(){
    cout<<"ReadConfigInfo constructor"<<endl;
}
ReadConfigInfo::~ReadConfigInfo(){
    cout<<"ReadConfigInfo destructor"<<endl;
}
map<string, string> ReadConfigInfo::getHostInfo(){
    map<string, string> info;
    CMarkup xml;
    string configXmlPath = getProjectRootPath() + "/config.xml";

    if(!xml.Load(configXmlPath)){
        cout<<"load config.xml failure";
        return info;
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
        cout<<stra[i]<<endl;
        bool flag= xml.FindElem(stra[i]);
        if (!flag){
            continue;
        }
        string ciValue = xml.GetAttrib("value");
        info.insert(make_pair(stra[i], ciValue));
        xml.OutOfElem();
    }
    return info;
}

map<string, string> ReadConfigInfo::getFuncInfo(string funcName){
    map<string, string> info;
    CMarkup xml;
    string configXmlPath = getProjectRootPath() + "/config.xml";

    if(!xml.Load(configXmlPath)){
        cout<<"load config.xml failure";
        return info;
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

                if(strName == "startTime" ){
                    time_t t= TimeTransfer::convertTimeStr2TimeStamp(strAttrib);
                    std::stringstream st;
                    st<<std::put_time(std::localtime(&t),"%F %X");
                    st.str();
                }
                info.insert(make_pair(strName, strAttrib));
                // do something with strName, strAttrib
            }
            break;
        }
    }
    return info;
}
string ReadConfigInfo::getProjectRootPath(){
    char *path = getenv("RESOURCE_DIR");
    if (path != nullptr) {
        printf("project 's absolute path in ENV:%s\n", path);
        return path;
    }

    // not find in ENV
    string project_path = PROJECT_ROOT_PATH;
    string resource_path = project_path;
    printf("project 's absolute path in CMake:%s\n", resource_path.c_str());
    return resource_path;
}