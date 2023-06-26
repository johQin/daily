//
// Created by buntu on 2023/6/21.
//

#ifndef RAINWARN_DATABS_H
#define RAINWARN_DATABS_H
#include <string>
#include<map>
#include<list>
#include<vector>

using namespace std;
typedef int (*callback)(list<vector<string>>);
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
