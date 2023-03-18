// 简易线程池
#include"threadpoolsimple"
#include<unistd.h>
#include<fcntl.h>
ThreadPool *thrPool = NULL;
int beginnum = 1000;
void *thrRun(void *arf){
    ThreadPool *pool = (ThreadPool *) arg;
    int taskpos = 0;//任务位置
    
}