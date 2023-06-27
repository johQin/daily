//
// Created by buntu on 2023/6/21.
//

#ifndef RAINWARN_DATABS_H
#define RAINWARN_DATABS_H
#include <string>
#include<map>
#include<list>
#include<vector>
#include <functional>
using namespace std;

// typedef std::function<void(int)> callback;
using callback = function<void(list<vector<string>> &list)>; //可以这样写，更直观
class DataBS{
private:
    map<string, string> dbInfoMap;
public:
    DataBS();
    ~DataBS();
public:
    int query(string sqlStr,callback call_fun);
    void queryCallback();
    void dbHello();
};
#endif //RAINWARN_DATABS_H
