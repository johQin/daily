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
#include "ThreadPool.h"

int CreateLogServer(CProcess* proc)
{
    //printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
    CLoggerServer server;
    printf("\n");
    int ret = server.Start();
    if (ret != 0) {
        printf("%s(%d):<%s> pid=%d errno:%d msg:%s ret:%d\n",
               __FILE__, __LINE__, __FUNCTION__, getpid(), errno, strerror(errno), ret);
    }
    int fd = 0;
    while (true) {
        ret = proc->RecvFD(fd);
        printf("%s(%d):<%s> fd=%d\n", __FILE__, __LINE__, __FUNCTION__, fd);
        // 子进程在收到fd为-1时，跳出去，关闭服务
        if (fd <= 0)break;
    }
    ret = server.Close();
    printf("%s(%d):<%s> ret=%d\n", __FILE__, __LINE__, __FUNCTION__, ret);
    return 0;
}
int main() {
    CProcess proclog;
    proclog.SetEntryFunction(CreateLogServer, &proclog);
    int ret = proclog.CreateSubProcess();
    if (ret != 0) {
        printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
        return -1;
    }
    logTest();

    CThreadPool pool;
    ret = pool.Start(4);
    printf("%s(%d):<%s> ret=%d\n", __FILE__, __LINE__, __FUNCTION__, ret);
    ret = pool.AddTask(log,"info");
    printf("%d",ret);

    getchar();
    proclog.SendFD(-1);
    return 0;
}
