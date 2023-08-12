//
// Created by buntu on 2023/8/11.
//

#ifndef EPLAYER_EPLAYER_H
#define EPLAYER_EPLAYER_H
#include "Logger.h"
#include "CServer.h"
#include <map>
/*
* 1. 客户端的地址问题，通过server发送的fd，以前是携带冗余，现在是携带地址过来
* 2. 连接回调的参数问题，传客户端，
* 3. 接收回调的参数问题，传客户端和数据
*/
#define ERR_RETURN(ret, err) if(ret!=0){TRACEE("ret= %d errno = %d msg = [%s]", ret, errno, strerror(errno));return err;}

#define WARN_CONTINUE(ret) if(ret!=0){TRACEW("ret= %d errno = %d msg = [%s]", ret, errno, strerror(errno));continue;}

class CEdoyunPlayerServer : public CBusiness
{
public:
    CEdoyunPlayerServer(unsigned count) :CBusiness() {
        m_count = count;
    }
    ~CEdoyunPlayerServer() {
        m_epoll.Close();
        m_pool.Close();
        for (auto it : m_mapClients) {
            if (it.second) {
                delete it.second;
            }
        }
        m_mapClients.clear();
    }
    // 消费来自server进程通过pipe发来的socket，并且将其添加到epoll上，供线程threadFunc去消费
    virtual int BusinessProcess(CProcess* proc) {
        // 用来占位的
        using namespace std::placeholders;
        int ret = 0;
        ret = setConnectedCallback(&CEdoyunPlayerServer::Connected, this, _1);
        ERR_RETURN(ret, -1);

        ret = setRecvCallback(&CEdoyunPlayerServer::Received, this, _1, _2);
        ERR_RETURN(ret, -2);

        ret = m_epoll.Create(m_count);
        ERR_RETURN(ret, -1);

        ret = m_pool.Start(m_count);
        ERR_RETURN(ret, -2);

        for (unsigned i = 0; i < m_count; i++) {
            ret = m_pool.AddTask(&CEdoyunPlayerServer::ThreadFunc, this);
            ERR_RETURN(ret, -3);
        }

        int sock = 0;
        sockaddr_in addrin;
        // 从server端收socket
        while (m_epoll != -1) {
            ret = proc->RecvSocket(sock, &addrin);
            if (ret < 0 || (sock == 0))break;
            CSocketBase* pClient = new CSocket(sock);
            if (pClient == NULL)continue;
            ret = pClient->Init(CSockParam(&addrin, SOCK_ISIP));
            WARN_CONTINUE(ret);
            ret = m_epoll.Add(sock, EpollData((void*)pClient));
            if (m_connectedcallback) {
                (*m_connectedcallback)(pClient);
            }
            WARN_CONTINUE(ret);
        }
        return 0;
    }
private:
    int Connected(CSocketBase* pClient) {
        return 0;
    }
    int Received(CSocketBase* pClient, const Buffer& data) {
        return 0;
    }
private:
    // 消费网络发过来的消息
    int ThreadFunc() {
        int ret = 0;
        EPEvents events;
        while (m_epoll != -1) {
            ssize_t size = m_epoll.WaitEvents(events);
            if (size < 0)break;
            if (size > 0) {
                for (ssize_t i = 0; i < size; i++)
                {
                    if (events[i].events & EPOLLERR) {
                        break;
                    }
                    else if (events[i].events & EPOLLIN) {
                        CSocketBase* pClient = (CSocketBase*)events[i].data.ptr;
                        if (pClient) {
                            Buffer data;
                            ret = pClient->Recv(data);
                            WARN_CONTINUE(ret);
                            if (m_recvcallback) {
                                (*m_recvcallback)(pClient, data);
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
    std::map<int, CSocketBase*> m_mapClients;
    CThreadPool m_pool;
    // 线程池线程个数，epoll的初始长度
    unsigned m_count;
};
#endif //EPLAYER_EPLAYER_H
