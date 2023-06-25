//
// Created by buntu on 2023/6/25.
//

#ifndef RAINWARN_UTILS_H
#define RAINWARN_UTILS_H
#include <stdio.h>
#include <string>
#include <time.h>
class TimeTransfer{
public:
    static time_t convertTimeStr2TimeStamp(string timeStr){
        struct tm timeinfo;
        strptime(timeStr.c_str(), "%Y-%m-%d %H:%M:%S",  &timeinfo);
        time_t timeStamp = mktime(&timeinfo);
        printf("timeStamp=%ld\n",timeStamp);
        return timeStamp;
    };
    static std::string convertTimeStamp2TimeStr(time_t timeStamp){
        struct tm *timeinfo = nullptr;
        char buffer[80];
        timeinfo = localtime(&timeStamp);
        strftime(buffer,80,"%Y-%m-%d %H:%M:%S",timeinfo);
        printf("%s\n",buffer);
        return std::string(buffer);
    }
};

#endif //RAINWARN_UTILS_H
