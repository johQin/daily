//
// Created by buntu on 2023/6/21.
//

#ifndef RAINWARN_RAIN_H
#define RAINWARN_RAIN_H
#include <map>
#include<list>
#include<vector>
#include<string>

class Rain {
private:
    float rainfall;
    std::map<std::string,std::string> funInfoMap;
public:
    Rain();
//    ~Rain();
public:
    void rainWarn();
    int rainDataHandleCallback(std::list<std::vector<std::string>> &list);
    int personInMineHandleCallback(std::list<std::vector<std::string>> &list);
};
#endif //RAINWARN_RAIN_H
