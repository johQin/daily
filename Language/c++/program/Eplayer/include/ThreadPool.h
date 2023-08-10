//
// Created by buntu on 2023/8/7.
//

#ifndef EPLAYER_THREADPOOL_H
#define EPLAYER_THREADPOOL_H
#include "Epoll.h"
#include "Thread.h"
#include "Function.h"
#include "Socket.h"

class CThreadPool
{
public:
    CThreadPool() {
        m_server = NULL;
        // 可以精确到ns
        timespec tp = { 0,0 };
        clock_gettime(CLOCK_REALTIME, &tp);     // #include<time.h>
        char* buf = NULL;
        asprintf(&buf, "%d.%d.sock", tp.tv_sec % 100000, tp.tv_nsec % 1000000);
        if (buf != NULL) {
            m_path = buf;
            free(buf);
        }       //在start函数里面还需要判断m_path是否初始化成功
        usleep(1);      // 这里是为了避免多个线程池初始化，而造成m_path重复
    }
    ~CThreadPool() {
        Close();
    }
    CThreadPool(const CThreadPool&) = delete;
    CThreadPool& operator=(const CThreadPool&) = delete;
public:
    // 开启服务，开启多个线程，去epoll中，消费任务
    // 在Start的外层，如果返回一个负数，记得手动Close一下
    int Start(unsigned count) {
        int ret = 0;
        if (m_server != NULL)return -1;//已经初始化了
        if (m_path.size() == 0)return -2;//构造函数失败！！！

        m_server = new CSocket();
        if (m_server == NULL)return -3;
        ret = m_server->Init(CSockParam(m_path, SOCK_ISSERVER));
        if (ret != 0)return -4;
        ret = m_epoll.Create(count);
        if (ret != 0)return -5;
        ret = m_epoll.Add(*m_server, EpollData((void*)m_server));
        if (ret != 0)return -6;
        // 线程id的vector
        m_threads.resize(count);
        // 对vector<CThread *> m_threads进行初始化，因为列表里放的是指针，所以必须初始化否则就是野指针
        for (unsigned i = 0; i < count; i++) {
            m_threads[i] = new CThread(&CThreadPool::TaskDispatch, this);
            if (m_threads[i] == NULL)return -7;
            ret = m_threads[i]->Start();
            if (ret != 0)return -8;
        }
        return 0;
    }
    void Close() {
        m_epoll.Close();
        // 短时操作，迅速锁住if
        if (m_server) {
            CSocketBase* p = m_server;
            m_server = NULL;
            delete p;
        }
        for (auto thread : m_threads)
        {
            if (thread)delete thread;
        }
        m_threads.clear();
        // int unlink(const char *pathname);    删除所给参数指定的文件。
        // 使用unlink函数删除文件的时候，只会删除 目录项 ，并且将inode节点的硬链接数目减一而已，并不一定会释放inode节点。
        // 在有进程打开此文件的情况下，则暂时不会删除，直到所有打开该文件的进程都结束时文件就会被删除。
        unlink(m_path);
    }
    // 客户端添加任务，通过本地socket的方式向服务器发送任务，传递的是一个函数（任务）
    template<typename _FUNCTION_, typename... _ARGS_>
    int AddTask(_FUNCTION_ func, _ARGS_... args) {
        // 如果同一个线程里多次调用addTask，那么都会共用一个客户端
        static thread_local CSocket client;
        int ret = 0;
        // 如果client == -1，那么就证明客户端没有初始化
        if (client == -1) {
            ret = client.Init(CSockParam(m_path, 0));
            if (ret != 0)return -1;
            // 初始化后一定要link
            ret = client.Link();
            if (ret != 0)return -2;
        }
        CFunctionBase* base = new CFunction< _FUNCTION_, _ARGS_...>(func, args...);
        if (base == NULL)return -3;

        // 将base拷贝一份，给data。
        Buffer data(sizeof(base));
        memcpy(data, &base, sizeof(base));

        // 通过客户端发送任务函数的指针
        ret = client.Send(data);
        if (ret != 0) {     // 失败
            delete base;
            return -4;
        }
        return 0;
    }
private:
    // 消费epoll中的连接，拿到任务（客户端发送过来是一个函数）之后，执行这个任务
    int TaskDispatch() {
        while (m_epoll != -1) {
            EPEvents events;
            int ret = 0;
            ssize_t esize = m_epoll.WaitEvents(events);
            if (esize > 0) {
                for (ssize_t i = 0; i < esize; i++) {
                    // 读事件
                    if (events[i].events & EPOLLIN) {

                        CSocketBase* pClient = NULL;

                        // lfd
                        if (events[i].data.ptr == m_server) {//客户端请求连接

                            ret = m_server->Link(&pClient);
                            if (ret != 0)continue;
                            ret = m_epoll.Add(*pClient, EpollData((void*)pClient));
                            if (ret != 0) {
                                delete pClient;
                                continue;
                            }
                        }

                        // cfd
                        else {//客户端的数据来了
                            pClient = (CSocketBase*)events[i].data.ptr;
                            if (pClient) {
                                // 接收任务
                                CFunctionBase* base = NULL;
                                Buffer data(sizeof(base));
                                ret = pClient->Recv(data);
                                if (ret <= 0) {
                                    m_epoll.Del(*pClient);
                                    delete pClient;
                                    continue;
                                }
                                memcpy(&base, (char*)data, sizeof(base));
                                if (base != NULL) {
                                    (*base)();
                                    delete base;
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }
private:
    CEpoll m_epoll;
    // 容器里面的对象一定要有默认构造函数和拷贝构造函数，否则不能放在容器里面
    // 由于CThread没有赋值构造函数，所以只能用对应的指针来代替
    std::vector<CThread*> m_threads;
    CSocketBase* m_server;
    Buffer m_path;
};
#endif //EPLAYER_THREADPOOL_H
