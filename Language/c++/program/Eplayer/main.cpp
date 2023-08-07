#include <iostream>
#include "include/MultiProcess.h"
#include <iostream>
#include<unistd.h>
#include<functional>
#include<memory.h>
#include<sys/socket.h>  //socket_pair
#include<sys/types.h>   //类型的定义
#include<sys/stat.h>    //system的状态值
#include<fcntl.h>       //文件操作
#include<signal.h>
#include<Logger.h>

int main() {
//    testMultiProcess();
    logTest();
//    char buffer[] = "hello LogTest! 神奇的世界，你好";
//    DUMPD((void*)buffer, (size_t)sizeof(buffer));
    return 0;
}
