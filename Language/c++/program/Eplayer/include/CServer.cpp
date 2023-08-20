#include "CServer.h"
#include "Logger.h"

CServer::CServer()
{
    m_server = NULL;
    m_business = NULL;
}

int CServer::Init(CBusiness* business, const Buffer& ip, short port)
{
    int ret = 0;
    if (business == NULL)return -1;
    m_business = business;
    ret = m_process.SetEntryFunction(&CBusiness::BusinessProcess, m_business, &m_process);
    if (ret != 0)return -2;
    // 创建子进程
    ret = m_process.CreateSubProcess();
    if (ret != 0)return -3;
    // 创建线程池
    ret = m_pool.Start(2);
    if (ret != 0)return -4;

    // 创建一个epoll，用于接收网络套接字
    ret = m_epoll.Create(2);
    if (ret != 0)return -5;

    // lfd
    m_server = new CSocket();
    if (m_server == NULL)return -6;
    ret = m_server->Init(CSockParam(ip, port, SOCK_ISSERVER | SOCK_ISIP | SOCK_ISREUSE));
    if (ret != 0)return -7;

    // 添加lfd到epoll中
    ret = m_epoll.Add(*m_server, EpollData((void*)m_server));
    if (ret != 0)return -8;

    // 用消费线程池中的线程，去执行ThreadFunc，而ThreadFunc中去消费网络连接
    for (size_t i = 0; i < m_pool.Size(); i++) {
        ret = m_pool.AddTask(&CServer::ThreadFunc, this);
        if (ret != 0)return -9;
    }
    return 0;
}
// 无限制等待
int CServer::Run()
{
    while (m_server != NULL) {
        usleep(10);
    }
    return 0;
}

int CServer::Close()
{
    if (m_server) {
        CSocketBase* sock = m_server;
        m_server = NULL;
        m_epoll.Del(*sock);
        delete sock;
    }
    m_epoll.Close();
    m_process.SendFD(-1);
    m_pool.Close();
    return 0;
}

int CServer::ThreadFunc()
{
    TRACEI("epoll %d server %p", (int)m_epoll, m_server);
    int ret = 0;
    EPEvents events;
    while ((m_epoll != -1) && (m_server != NULL)) {
        ssize_t size = m_epoll.WaitEvents(events, 500);
        if (size < 0)break;
        if (size > 0) {
            TRACEI("size=%d event %08X", size, events[0].events);
            for (ssize_t i = 0; i < size; i++)
            {
                // 这里面不会去处理客户端与服务器之间的收发
                if (events[i].events & EPOLLERR) {
                    break;
                }
                else if (events[i].events & EPOLLIN) {
                    if (m_server) {
                        CSocketBase* pClient = NULL;
                        ret = m_server->Link(&pClient);
                        if (ret != 0)continue;
                        // 向子进程发送套接字
                        ret = m_process.SendSocket(*pClient, *pClient);
                        TRACEI("SendSocket %d", ret);
                        if (ret != 0) {
                            TRACEE("send client %d failed!", (int)*pClient);
                            continue;
                        }
                        delete pClient;
                        pClient = NULL;
                    }
                }
            }
        }
    }
    TRACEI("服务器已停止");
    return 0;
}
