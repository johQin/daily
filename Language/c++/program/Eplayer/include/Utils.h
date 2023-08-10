//
// Created by buntu on 2023/8/10.
//

#ifndef EPLAYER_UTILS_H
#define EPLAYER_UTILS_H
#include <string>
#include <time.h>
class TimeTransfer{
public:
    TimeTransfer();
    virtual ~TimeTransfer();
public:
    static time_t convertTimeStr2TimeStamp(std::string timeStr);
    static std::string convertTimeStamp2TimeStr(time_t timeStamp);
};
TimeTransfer::TimeTransfer() {};
TimeTransfer::~TimeTransfer() {};
time_t TimeTransfer::convertTimeStr2TimeStamp(std::string timeStr= ""){
    time_t timeStamp;
    if(timeStr.length() == 0){
        time(&timeStamp);
        return timeStamp;
    }
    struct tm timeinfo;
    strptime(timeStr.c_str(), "%Y-%m-%d %H:%M:%S",  &timeinfo);
    timeinfo.tm_isdst = -1;
    timeStamp = mktime(&timeinfo);
    return timeStamp;
};
std::string TimeTransfer::convertTimeStamp2TimeStr(time_t timeStamp = 0){
    struct tm *timeinfo = nullptr;
    if(timeStamp == 0 ){
        time(&timeStamp);
    }
    timeinfo = localtime(&timeStamp);
    char buffer[80];
    strftime(buffer,80,"%Y-%m-%d %H:%M:%S",timeinfo);
    return std::string(buffer);
}

#endif //EPLAYER_UTILS_H
