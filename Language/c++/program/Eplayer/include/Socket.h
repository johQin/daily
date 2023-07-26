//
// Created by buntu on 2023/7/26.
//

#ifndef EPLAYER_SOCKET_H
#define EPLAYER_SOCKET_H
#include<unistd.h>      //数据类型
#include<sys/socket.h>  //套接字
#include<sys/un.h>      //本地套接字
#include<netinet/in.h>  //网络套接字，
#include<arpa/inet.h>   //网络一些api接口
#include <string>

// 继承字符串类
class Buffer :public std::string
{
public:
    // 提供一个默认构造函数
    Buffer() :std::string() {}
    // 按照空间大小构造缓冲区。构造Buffer的时候，会触发父类string的构造函数，并且调用resize函数调整空间大小
    Buffer(size_t size) :std::string() { resize(size); }

    // 继承std::string 转字符常量指针的用法，这就是为什么要继承的原因
    // 可以转换Buffer 为char * 指针
    operator char* () { return (char*)c_str(); }
    // 可以转换const Buffer 为 char * 指针
    operator char* () const { return (char*)c_str(); }
    // 可以转换const Buffer 为 const char * 指针
    operator const char* () const { return c_str(); }
};

enum SocketAttr {
    SOCK_ISSERVER = 1,//是否服务器 1服务器 0客户端
    SOCK_ISBLOCK = 2,//是否阻塞 1阻塞 0非阻塞
};

class CSocketParam {
public:
    CSocketParam() {
        bzero(&addr_in, sizeof(addr_in));
        bzero(&addr_un, sizeof(addr_un));
        port = -1;
        attr = 0;
    }
    // 网络套接字的构造器
    CSocketParam(const Buffer& ip, short port, int attr) {
        this->ip = ip;
        this->port = port;
        this->attr = attr;
        addr_in.sin_family = AF_INET;
        addr_in.sin_port = port;
        addr_in.sin_addr.s_addr = inet_addr(ip);        // 这里就用上了Buffer的类型转换函数，自动将buffer转换为const char *
    }
    // 本地套接字的构造器，不需要给定port
    CSocketParam(const Buffer& path, int attr) {
        //用ip来记录路径
        ip = path;
        addr_un.sun_family = AF_UNIX;
        strcpy(addr_un.sun_path, path);
        this->attr = attr;
    }
    ~CSocketParam() {}
    // 拷贝构造函数
    CSocketParam(const CSocketParam& param) {
        ip = param.ip;
        port = param.port;
        attr = param.attr;
        memcpy(&addr_in, &param.addr_in, sizeof(addr_in));
        memcpy(&addr_un, &param.addr_un, sizeof(addr_un));
    }
public:
    // 重载赋值运算符
    CSocketParam& operator=(const CSocketParam& param) {
        if (this != &param) {
            ip = param.ip;
            port = param.port;
            attr = param.attr;
            memcpy(&addr_in, &param.addr_in, sizeof(addr_in));
            memcpy(&addr_un, &param.addr_un, sizeof(addr_un));
        }
        return *this;
    }
    sockaddr* addrin() { return (sockaddr*)&addr_in; }
    sockaddr* addrun() { return (sockaddr*)&addr_un; }
public:
    // 套接字的地址
    sockaddr_in addr_in;    //网络
    sockaddr_un addr_un;    //本地
    //ip
    Buffer ip;
    // 端口
    short port;
    //参考SocketAttr
    int attr;
};

//socket 的接口类，到时候具体实现由子类实现
class CSocketBase
{
public:
    //通过父类指针释放子类空间
    virtual ~CSocketBase() {
        m_status = 3;
        if (m_socket != -1) {
            int fd = m_socket;
            m_socket = -1;
            // close需要花多长时间不确认
            close(fd);
        }
    }
public:
    // 纯虚函数，代表这一定是一个接口类，不一一实现，就无法构造对象

    //初始化，服务端包括套接字创建socket，bind，listen。客户端只需要套接字创建
    virtual int Init(const CSocketParam& param) = 0;
    //连接，服务端accept（服务端会有一个客户端参数），客户端connect（参数默认为NULL），对于UDP来讲，这里可以忽略
    virtual int Link(CSocketBase** pClient = NULL) = 0;
    // 发送数据
    virtual int Send(const Buffer& data) = 0;
    // 接收数据
    virtual int Recv(Buffer& data) = 0;
    //关闭连接
    virtual int Close() = 0;
protected:
    //套接字描述符
    int m_socket;
    //状态记录，永远不要高估使用者的智商，使用者可能跳过init，link，而直接发起收发，我们可以通过此字段来，判断socket的执行阶段，
    // 0未初始化，1已初始化 2已连接 3已关闭
    int m_status;
};
#endif //EPLAYER_SOCKET_H
