//
// Created by buntu on 2023/6/15.
//

#ifndef EPLAYER_DATABASEHELPER_H
#define EPLAYER_DATABASEHELPER_H


class CDatabaseClient {
public:
    CDatabaseClient(const CDatabaseClient&) = delete; //不允许拷贝构造
    CDatabaseClient& operator=(const CDatabaseClient &) = delete;   //不允许赋值
public:
    CDatabaseClient(){}
    virtual ~CDatabaseClient(){}
};


#endif //EPLAYER_DATABASEHELPER_H
