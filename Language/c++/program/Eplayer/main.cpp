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

int main() {
    testMultiProcess();
    return 0;
}
