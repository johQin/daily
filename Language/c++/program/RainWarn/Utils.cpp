//
// Created by buntu on 2023/6/26.
//
//
// Created by buntu on 2023/6/25.
//

#include <stdio.h>
#include <string>
#include <time.h>
#include "Utils.h"
#include<iostream>
#include <cstdio>
//union TimeTransfer::TimeStamp{
//time_t stamp_t;
//long int stamp_int;
//std::string stamp_str;
//} tsu;
TimeTransfer::TimeTransfer() {};
TimeTransfer::~TimeTransfer() {};
time_t TimeTransfer::convertTimeStr2TimeStamp(std::string timeStr){
    struct tm timeinfo;
    strptime(timeStr.c_str(), "%Y-%m-%d %H:%M:%S",  &timeinfo);
    time_t timeStamp = mktime(&timeinfo);
    printf("timeStamp=%ld\n",timeStamp);
    return timeStamp;
};
std::string TimeTransfer::convertTimeStamp2TimeStr(time_t timeStamp){
    struct tm *timeinfo = nullptr;
    char buffer[80];
    timeinfo = localtime(&timeStamp);
    strftime(buffer,80,"%Y-%m-%d %H:%M:%S",timeinfo);
    printf("%s\n",buffer);
    return std::string(buffer);
}

