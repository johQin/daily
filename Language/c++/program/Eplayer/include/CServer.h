//
// Created by buntu on 2023/8/10.
//

#ifndef EPLAYER_CSERVER_H
#define EPLAYER_CSERVER_H

#include "Socket.h"
#include "Epoll.h"
#include "ThreadPool.h"
#include "Process.h"

template<typename _FUNCTION_, typename... _ARGS_>
class CConnectedFunction :public CFunctionBase
{
public:
    CConnectedFunction(_FUNCTION_ func, _ARGS_... args)
            :m_binder(std::forward<_FUNCTION_>(func), std::forward<_ARGS_>(args)...)
    {}
    virtual ~CConnectedFunction() {}
    virtual int operator()(CSocketBase* pClient) {
        return m_binder(pClient);
    }
    typename std::_Bindres_helper<int, _FUNCTION_, _ARGS_...>::type m_binder;
};

template<typename _FUNCTION_, typename... _ARGS_>
class CReceivedFunction :public CFunctionBase
{
public:
    CReceivedFunction(_FUNCTION_ func, _ARGS_... args)
            :m_binder(std::forward<_FUNCTION_>(func), std::forward<_ARGS_>(args)...)
    {}
    virtual ~CReceivedFunction() {}
    virtual int operator()(CSocketBase* pClient, const Buffer& data) {
        return m_binder(pClient, data);
    }
    typename std::_Bindres_helper<int, _FUNCTION_, _ARGS_...>::type m_binder;
};

// 业务模块接口层，主要是将通信层和业务模块解耦
class CBusiness
{
public:
    CBusiness():m_connectedcallback(NULL), m_recvcallback(NULL){}
    virtual int BusinessProcess(CProcess* proc) = 0;
    template<typename _FUNCTION_, typename... _ARGS_>
    int setConnectedCallback(_FUNCTION_ func, _ARGS_... args) {
        m_connectedcallback = new CConnectedFunction< _FUNCTION_, _ARGS_...>(func, args...);
        if (m_connectedcallback == NULL)return -1;
        return 0;
    }
    template<typename _FUNCTION_, typename... _ARGS_>
    int setRecvCallback(_FUNCTION_ func, _ARGS_... args) {
        m_recvcallback = new CReceivedFunction< _FUNCTION_, _ARGS_...>(func, args...);
        if (m_recvcallback == NULL)return -1;
        return 0;
    }
protected:
    CFunctionBase* m_connectedcallback;
    CFunctionBase* m_recvcallback;
};

class CServer
{
public:
    CServer();
    ~CServer() { Close(); }
    CServer(const CServer&) = delete;   // 服务器是不能复制的
    CServer& operator=(const CServer&) = delete;
public:
    int Init(CBusiness* business, const Buffer& ip = "127.0.0.1", short port = 9999);
    int Run();
    int Close();
private:
    int ThreadFunc();
private:
    CThreadPool m_pool;
    CSocketBase* m_server;
    // 这个epoll主要是用来接入客户端的
    CEpoll m_epoll;
    CProcess m_process;
    CBusiness* m_business;      //业务模块 需要我们手动delete
};
#endif //EPLAYER_CSERVER_H
