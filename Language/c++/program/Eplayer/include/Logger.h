//
// Created by buntu on 2023/7/30.
//

// 目前该头文件存在的疑点
// 1.logger.h 的ThreadFunc里，当客户端收到的信息长度小于0的时候，为什么不从epoll里面m_epoll.Del删除，而仅仅是delete mapClients对应的客户端

#ifndef EPLAYER_LOGGER_H
#define EPLAYER_LOGGER_H

#include "Thread.h"
#include "Epoll.h"
#include "Socket.h"
#include <list>
#include <sys/timeb.h>
#include <stdarg.h>
#include <sstream>          //stringstream
#include <sys/stat.h>       //mkdir
#include<stdio.h>

enum LogLevel {
    LOG_INFO,
    LOG_DEBUG,
    LOG_WARNING,
    LOG_ERROR,
    LOG_FATAL
};

class LogInfo {
public:
    // 针对可变参数宏，TRACEI系列
    LogInfo(
            const char* file, int line, const char* func,
            pid_t pid, pthread_t tid, int level,
            const char* fmt, ...);
    // 针对LOG系列宏
    LogInfo(
            const char* file, int line, const char* func,
            pid_t pid, pthread_t tid, int level);
    // 针对DUMP系列宏
    LogInfo(const char* file, int line, const char* func,
            pid_t pid, pthread_t tid, int level,
            void* pData, size_t nSize);

    ~LogInfo();
    // 类型转换函数
    operator Buffer()const {
        return m_buf;
    }
    // 重载输入操作运算符，调用时，类型可以自动推导
    template<typename T>
    LogInfo& operator<<(const T& data) {
        std::stringstream stream;
        stream << data;
        //printf("%s(%d):[%s][%s]\n", __FILE__, __LINE__, __FUNCTION__, stream.str().c_str());
        m_buf += stream.str().c_str();
        //printf("%s(%d):[%s][%s]\n", __FILE__, __LINE__, __FUNCTION__, (char*)m_buf);
        return *this;
    }
private:
    bool bAuto;     //默认是false 流式日志则为true
    Buffer m_buf;
};

class CLoggerServer
{
public:
    CLoggerServer() :
            m_thread(&CLoggerServer::ThreadFunc, this)
    {
        m_server = NULL;

        char curpath[256] = "";
        getcwd(curpath, sizeof(curpath));
        m_path = curpath;
        m_path += "/log/" + GetTimeStr() + ".log";
    }
    ~CLoggerServer() {
        Close();
    }
public:
    // 不能拷贝
    CLoggerServer(const CLoggerServer&) = delete;
    // 不能赋值
    CLoggerServer& operator=(const CLoggerServer&) = delete;
public:
    //日志服务器的启动
    int Start() {
        // 如果已经start过了，那么就不需要再start了
        if (m_server != NULL)return -1;

        // 判断文件或文件夹是否可读可写，int access (const char *__name, int __type)
        if (access("log", W_OK | R_OK) != 0) {
            // 用户的读/写，用户组的读/写，其他写
            mkdir("log", S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
        }
        // w+，不存在就创建
        m_file = fopen(m_path, "w+");
        if (m_file == NULL)return -2;

        int ret = m_epoll.Create(1);
        if (ret != 0)return -3;


        // 生成首个监听套接字，作监听
        // lfd
        m_server = new CSocket();
        if (m_server == NULL) {
            Close();
            return -4;
        }
            // 如果是服务器：套接字创建、bind、listen
        ret = m_server->Init(CSockParam("./log/server.sock", (int)SOCK_ISSERVER));
        if (ret != 0) {
            Close();
            return -5;
        }

        // 将lfd添加进去
        ret = m_epoll.Add(*m_server, EpollData((void*)m_server), EPOLLIN | EPOLLERR);
        if (ret != 0) {
            Close();
            return -6;
        }

        //日志服务器的启动，m_thread已在构造器中初始化
        ret = m_thread.Start();
        if (ret != 0) {
            printf("%s(%d):<%s> pid=%d errno:%d msg:%s ret:%d\n",
                   __FILE__, __LINE__, __FUNCTION__, getpid(), errno, strerror(errno), ret);
            Close();
            return -7;
        }

        return 0;
    }
    int Close() {
        if (m_server != NULL) {
            CSocketBase* p = m_server;
            m_server = NULL;
            delete p;
        }
        m_epoll.Close();
        m_thread.Stop();
        return 0;
    }
    //其他非日志进程的进程和线程发送日志信息的函数
    static void Trace(const LogInfo& info) {
        int ret = 0;
        // static 局部静态变量，初始化后，每次调用函数后，不会再初始化
        // thread_local 限定static在线程内，每一个线程调用此函数只会初始化一次
        static thread_local CSocket client;

        if (client == -1) {
            ret = client.Init(CSockParam("./log/server.sock", 0));
            if (ret != 0) {
#ifdef _DEBUG
                printf("%s(%d):[%s]ret=%d\n", __FILE__, __LINE__, __FUNCTION__, ret);
#endif
                return;
            }
            printf("%s(%d):[%s]ret=%d client=%d\n", __FILE__, __LINE__, __FUNCTION__, ret, (int)client);
            ret = client.Link();
            printf("%s(%d):[%s]ret=%d client=%d\n", __FILE__, __LINE__, __FUNCTION__, ret, (int)client);
        }
        ret = client.Send(info);
        printf("%s(%d):[%s]ret=%d client=%d\n", __FILE__, __LINE__, __FUNCTION__, ret, (int)client);
    }
    static Buffer GetTimeStr() {
        Buffer result(128);
        timeb tmb;
        ftime(&tmb);        // ftime' is deprecated: Use gettimeofday or clock_gettime instead 'ftime' has been explicitly marked deprecated here
        tm* pTm = localtime(&tmb.time);

        // 拼接时间字符串，返回字符串长度
        int nSize = snprintf(result, result.size(),
                             "%04d-%02d-%02d_%02d-%02d-%02d.%03d",      //注意这里的字符串，用于给文件命名，所以不能用空格，等号，等符号
                             pTm->tm_year + 1900, pTm->tm_mon + 1, pTm->tm_mday,
                             pTm->tm_hour, pTm->tm_min, pTm->tm_sec,
                             tmb.millitm
        );
        result.resize(nSize);

        return result;
    }
private:
    // 用于从epoll中接受消息
    int ThreadFunc() {
//        printf("%s(%d):[%s] %d\n", __FILE__, __LINE__, __FUNCTION__, m_thread.isValid());
//        printf("%s(%d):[%s] %d\n", __FILE__, __LINE__, __FUNCTION__, (int)m_epoll);
//        printf("%s(%d):[%s] %p\n", __FILE__, __LINE__, __FUNCTION__, m_server);

        //
        EPEvents events;

        // map存放客户端套接字信息
        std::map<int, CSocketBase*> mapClients;

        while (m_thread.isValid() && (m_epoll != -1) && (m_server != NULL)) {
            // 监听有没有套接字就绪
            ssize_t ret = m_epoll.WaitEvents(events, 1);
//            printf("%s(%d):[%s] %d\n", __FILE__, __LINE__, __FUNCTION__, ret);

            // 没有套接字就绪
            if (ret < 0)break;

            // 有套接字就绪
            if (ret > 0) {
                ssize_t i = 0;
                for (; i < ret; i++) {

                    // 遇到错误
                    if (events[i].events & EPOLLERR) {
                        break;
                    }

                    // 如果是读事件变化
                    else if (events[i].events & EPOLLIN) {
                        //如果是lfd变化，说明有客户端来连接了
                        if (events[i].data.ptr == m_server) {

                            CSocketBase* pClient = NULL;

                            // Link 函数里面，pClient指向的对象是new出来的，所以后面需要delete
                            int r = m_server->Link(&pClient);
                            printf("%s(%d):[%s]ret=%d \n", __FILE__, __LINE__, __FUNCTION__, r);
                            if (r < 0) continue;

                            //将客户端放到epoll上面去
                            r = m_epoll.Add(*pClient, EpollData((void*)pClient), EPOLLIN | EPOLLERR);
                            printf("%s(%d):[%s]ret=%d \n", __FILE__, __LINE__, __FUNCTION__, r);
                            if (r < 0) {
                                delete pClient;
                                continue;
                            }

                            // 在添加到客户端套接字map的之前，先看看旧有map有没有该套接字，有的话就先删了，然后再添加
                            auto it = mapClients.find(*pClient);
                            if (it != mapClients.end()) {// 如果找不到，it将指向end，但it不等于NULL，it->second将直接无效，直接报错
                                //it->second != NULL
                                if (it->second)delete it->second;//delete it->second;
                            }
                            mapClients[*pClient] = pClient;
                            printf("%s(%d):[%s]ret=%d \n", __FILE__, __LINE__, __FUNCTION__, r);
                        }

                        //如果是cfd变化了，说明是客户端来消息了
                        else {
                            printf("%s(%d):[%s]ptr=%p \n", __FILE__, __LINE__, __FUNCTION__, events[i].data.ptr);
                            CSocketBase* pClient = (CSocketBase*)events[i].data.ptr;
                            if (pClient != NULL) {

                                Buffer data(1024 * 1024);   // 1M
                                int r = pClient->Recv(data);
                                printf("%s(%d):[%s]ret=%d \n", __FILE__, __LINE__, __FUNCTION__, r);

                                // r <= 0 说明数据已经读完，可以释放，但为什么不释放m_epoll这里有待考究
                                if (r <= 0) {
                                    mapClients[*pClient] = NULL;
                                    delete pClient;
                                }
                                // r > 0 说明有数据读
                                else {
                                    printf("%s(%d):[%s]data=%s \n", __FILE__, __LINE__, __FUNCTION__, (char*)data);
                                    WriteLog(data);
                                }
                            }
                        }
                    }
                }

                // 如果i，对每个套接字进行访问，i就会等于ret，如果不等于ret，那么就有问题
                if (i != ret) {
                    break;
                }
            }
        }

        // 跳出监听服务之后，删除所有new出来的pClient，防止内存泄漏
        for (auto it = mapClients.begin(); it != mapClients.end(); it++) {
            if (it->second) {
                delete it->second;
            }
        }
        // 清理掉map里的所有元素
        mapClients.clear();
        return 0;
    }
    // 在接收到日志信息之后，从这里写入文件
    void WriteLog(const Buffer& data) {
        if (m_file != NULL) {
            FILE* pFile = m_file;
            fwrite((char*)data, 1, data.size(), pFile);
            fflush(pFile);
// 在预处理（命令行参数上），加一个_DEBUG宏，就可以实现调试，-D "_DEBUG"
#ifdef _DEBUG
            printf("%s", (char*)data);
#endif
        }
    }
private:
    CThread m_thread;
    CEpoll m_epoll;
    CSocketBase* m_server;
    Buffer m_path;
    FILE* m_file;       // 日志的file文件
};

// 可变参数宏，__VA_ARGS__
#ifndef TRACE
#define TRACEI(...) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_INFO, __VA_ARGS__))
#define TRACED(...) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_DEBUG, __VA_ARGS__))
#define TRACEW(...) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_WARNING, __VA_ARGS__))
#define TRACEE(...) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_ERROR, __VA_ARGS__))
#define TRACEF(...) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_FATAL, __VA_ARGS__))

//LOGI<<"hello"<<"how are you";
#define LOGI LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_INFO)
#define LOGD LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_DEBUG)
#define LOGW LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_WARNING)
#define LOGE LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_ERROR)
#define LOGF LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_FATAL)

//内存导出
//00 01 02 65……  ; ...a……
//
#define DUMPI(data, size) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_INFO, data, size))
#define DUMPD(data, size) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_DEBUG, data, size))
#define DUMPW(data, size) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_WARNING, data, size))
#define DUMPE(data, size) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_ERROR, data, size))
#define DUMPF(data, size) CLoggerServer::Trace(LogInfo(__FILE__, __LINE__, __FUNCTION__, getpid(), pthread_self(), LOG_FATAL, data, size))
#endif


#endif //EPLAYER_LOGGER_H
