//
// Created by buntu on 2023/6/21.
//

#ifndef RAINWARN_RAIN_H
#define RAINWARN_RAIN_H
#include <map>
class Rain {
private:
    float rainfall;
    std::map<std::string,std::string> funInfoMap;
public:
    Rain();
//    ~Rain();
public:
    void rainWarn();

};
#endif //RAINWARN_RAIN_H
