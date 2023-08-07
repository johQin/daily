//
// Created by buntu on 2023/7/24.
//
#include "../../include/Process.h"
#include<stdio.h>
#include<errno.h>
#include "../../include/Logger.h"

int CreateClientServer(CProcess * proc){
    printf("子进程执行函数：CreateClientServer\n");
//    printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
    int fd = -1;

    sleep(1);
    int ret = proc->RecvFD(fd);
    printf("(%d):<%s>%d,%d\n",__LINE__, __FUNCTION__,ret, fd);
    printf("子进程fd：%d\n",fd);
    char buf[10] = "";
    lseek(fd, 0, SEEK_SET);
    read(fd, buf, sizeof(buf));
    printf("读到的内容：%s",buf);
    close(fd);
    return 0;
}
int testMultiProcess(){
    // 开启守护进程模式
    //CProcess::SwitchDaemon();
    CProcess procclient;
    //proclog.SetEntryFunction(CreateLogServer,&proclog);
    procclient.SetEntryFunction(CreateClientServer,&procclient);    // 可以自动推导类型
    procclient.CreateSubProcess();

    int fd = open("./test.txt",O_RDWR | O_CREAT | O_APPEND);
    if (fd ==-1) return -3;
    write(fd,"qqqkkkhhh",9);
    int ret =procclient.SendFD(fd);
    if (ret != 0) printf("errno:%d msg:%s\n", errno, strerror(errno));
    close(fd);
    printf("父进程\n");
    return 0;
}
int CreateLogServer(CProcess* proc)
{
    //printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
    CLoggerServer server;
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

int logTest()
{
    CProcess proclog;
    printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
    proclog.SetEntryFunction(CreateLogServer, &proclog);
    int ret = proclog.CreateSubProcess();
    if (ret != 0) {
        printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
        return -1;
    }


    //等待服务开启，100ms
    usleep(1000 * 100);

    //记录日志
    char buffer[] = "hello LogTest! 神奇的世界，你好";
    TRACEI("here is log %d %c %f %g %s 记录 log进程 风起云涌", 10, 'A', 1.0f, 2.0, buffer);
    DUMPD((void*)buffer, (size_t)sizeof(buffer));
    LOGE << 100 << " " << 'S' << " " << 0.12345f << " " << 1.23456789 << " " << buffer << " 编程";

    // 关闭日志服务器，主进程向子进程发送信息fd，当fd为-1的时候关闭服务
    proclog.SendFD(-1);
    getchar();
    return 0;
}