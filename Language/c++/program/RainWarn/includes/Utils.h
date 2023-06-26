//
// Created by buntu on 2023/6/25.
//

#ifndef RAINWARN_UTILS_H
#define RAINWARN_UTILS_H
#include <string>
#include <time.h>


//using namespace std;
class TimeTransfer{
public:
    TimeTransfer();
    virtual ~TimeTransfer();
public:
    static time_t convertTimeStr2TimeStamp(std::string timeStr);
    static std::string convertTimeStamp2TimeStr(time_t timeStamp);
};

#endif //RAINWARN_UTILS_H
