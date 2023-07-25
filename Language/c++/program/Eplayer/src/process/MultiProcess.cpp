//
// Created by buntu on 2023/7/24.
//
#include "../../include/Process.h"
#include<stdio.h>
#include<errno.h>

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