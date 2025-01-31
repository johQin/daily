//
// Created by buntu on 2023/6/21.
//

#ifndef RAINWARN_READCONFIGINFO_H
#define RAINWARN_READCONFIGINFO_H
#include<string>
#include<map>

using namespace std;
class ReadConfigInfo{
private:
    string xmlPath;
public:
    ReadConfigInfo();
    ~ReadConfigInfo();
public:
    int getFuncInfo(string funName,map<string,string> & info);
    int getHostInfo(map<string,string> & info);
    string getProjectRootPath();
};
#endif //RAINWARN_READCONFIGINFO_H
