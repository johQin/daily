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
#include<fcntl.h>       //file control 文件的打开、数据写入、数据读取、关闭文件的操作。
#include "Public.h"


enum SockAttr {
    // 这里的枚举值一定要是2的阶乘，2^0，2^1，2^2
    SOCK_ISSERVER = 1,//是否服务器 1服务器 0客户端
    SOCK_ISNONBLOCK = 2,//是否阻塞 1非阻塞 0阻塞
    SOCK_ISUDP = 4,//是否为UDP 1表示udp 0表示tcp
    SOCK_ISIP = 8,//是否为IP协议 1表示IP协议(网络套接字) 0表示本地套接字
    SOCK_ISREUSE = 16//是否重用地址,
};

class CSockParam {
public:
    CSockParam() {
        bzero(&addr_in, sizeof(addr_in));
        bzero(&addr_un, sizeof(addr_un));
        port = -1;
        attr = 0;   //默认是客户端，阻塞，tcp
    }
    // 网络套接字的构造器
    CSockParam(const Buffer& ip, short port, int attr) {
        this->ip = ip;
        this->port = port;
        this->attr = attr;
        addr_in.sin_family = AF_INET;
        addr_in.sin_port = htons(port);     // h host主机 n net 网络 s short，主机字节序转为网络字节序
        addr_in.sin_addr.s_addr = inet_addr(ip);        // 这里就用上了Buffer的类型转换函数，自动将buffer转换为const char *
    }
    // 本地套接字的构造器，不需要给定port
    CSockParam(const Buffer& path, int attr) {
        //用ip来记录路径
        ip = path;
        addr_un.sun_family = AF_UNIX;
        strcpy(addr_un.sun_path, path);
        this->attr = attr;
    }
    CSockParam(const sockaddr_in* addrin, int attr) {
        this->ip = ip;
        this->port = port;
        this->attr = attr;
        memcpy(&addr_in, addrin, sizeof(addr_in));
    }
    ~CSockParam() {}
    // 拷贝构造函数
    CSockParam(const CSockParam& param) {
        ip = param.ip;
        port = param.port;
        attr = param.attr;
        memcpy(&addr_in, &param.addr_in, sizeof(addr_in));
        memcpy(&addr_un, &param.addr_un, sizeof(addr_un));
    }
public:
    // 重载赋值运算符
    CSockParam& operator=(const CSockParam& param) {
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
    sockaddr_in addr_in;    //网络地址，有ip，port等
    sockaddr_un addr_un;    //本地地址，有path
    //ip
    Buffer ip;
    // 端口
    short port;
    //参考SockAttr
    int attr;
};

//socket 的接口类，到时候具体实现由子类实现
class CSocketBase
{
public:
    CSocketBase(){
        m_socket = -1;
        m_status = 0;   //未初始化
    }
    //通过父类指针释放子类空间
    virtual ~CSocketBase() {
       Close();
    }
public:
    // 纯虚函数，代表这一定是一个接口类，不一一实现，就无法构造对象

    //初始化，服务端包括套接字创建socket，bind，listen。客户端只需要套接字创建
    virtual int Init(const CSockParam& param) = 0;
    //连接，服务端accept（服务端会有一个客户端参数），客户端connect（参数默认为NULL），对于UDP来讲，这里可以忽略
    virtual int Link(CSocketBase** pClient = NULL) = 0;
    // 发送数据
    virtual int Send(const Buffer& data) = 0;
    // 接收数据
    virtual int Recv(Buffer& data) = 0;
    //关闭连接
    virtual int Close(){
        m_status = 3;
        if (m_socket != -1) {

            if (
                    ((m_param.attr & SOCK_ISIP) == 0)    // 先判断是不是本地套接字，必须是本地套接字
                    &&
                    (m_param.attr & SOCK_ISSERVER)      // 如果是本地套接字，所以只有一个套接文件，当客户端断调的时候不能删除这个本地套接字，只有当服务器关闭的时候，才删除本地套接字
               ){
                unlink(m_param.ip);
            }
            int fd = m_socket;
            m_socket = -1;
            // close需要花多长时间不确认
            close(fd);
        }
    };
    virtual operator int(){ return m_socket; }
    virtual operator int() const { return m_socket; }
    virtual operator const sockaddr_in*() const {return &m_param.addr_in;}
    virtual operator sockaddr_in*() {return &m_param.addr_in;}

protected:
    //套接字描述符
    int m_socket;
    //状态记录，永远不要高估使用者的智商，使用者可能跳过init，link，而直接发起收发，我们可以通过此字段来，判断socket的执行阶段，
    // 0未初始化，1已初始化 2已连接 3已关闭
    int m_status;
    //初始化参数
    CSockParam m_param;
};
class CSocket:public CSocketBase
{
public:
    // 子类构造函数，直接调用分类的构造函数
    CSocket() :CSocketBase() {}
    CSocket(int sock) :CSocketBase() {
        m_socket = sock;
    }
    //传递析构操作
    virtual ~CSocket() {
        Close();
    }
public:
    // 初始化
    // 服务器： 套接字创建、bind、listen
    // 客户端： 套接字创建
    virtual int Init(const CSockParam& param) {
        if (m_status != 0)return -1;
        m_param = param;

        // 判断是UDP还是TCP
        int type = (m_param.attr & SOCK_ISUDP) ? SOCK_DGRAM : SOCK_STREAM;

        // 创建套接字, 判断套接字是否创建
        // 正常情况下，套接字是-1的，因为这是第一个套接字lfd（监听套接字）
        // 但在link那里，通过m_socket(lfd) 接收 accept，创建一个新的已连接套接字（通信套接字，cfd），然后由这个cfd创建一个本地套接字，*pClient = new CSocket(cfd)，所以在这里是已经有m_socket,并且等于cfd
        if (m_socket == -1){
            // 在这里判断是需要网络套接字还是本地套接字
            if (param.attr & SOCK_ISIP)
                m_socket = socket(PF_INET, type, 0);    // 网络套接字
            else
                m_socket = socket(PF_LOCAL, type, 0);   // 本地套接字
        }
        else m_status = 2;      // 如果是accept来的cfd（客户端），它已经处于连接状态，所以要置为 2

        // 如果套接字创建失败，返回-2
        if (m_socket == -1)return -2;

        int ret = 0;
        // 端口复用最常用的用途应该是防止服务器重启时之前绑定的端口还未释放或者程序突然退出而系统没有释放端口。这种情况下如果设定了端口复用，则新启动的服务器进程可以直接绑定端口。如果没有设定端口复用，绑定会失败，提示ADDR已经在使用中——那只好等等再重试了，麻烦！
        if (m_param.attr & SOCK_ISREUSE) {
            int option = 1;
            ret = setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
            // SO_REUSEADDR ,不设置很可能会报：Address already in use
            if (ret == -1)return -7;
        }

        // 判断是不是服务器，如果是客户端这一段什么都不用干
        if (m_param.attr & SOCK_ISSERVER) {
            // 如果是服务器
            // 绑定，套接字、地址
            if (param.attr & SOCK_ISIP)
                ret = bind(m_socket, m_param.addrin(), sizeof(sockaddr_in));    //网络套接字：m_param.addrin()，返回网络套接字地址，网络套接字类型sockaddr_in
            else
                ret = bind(m_socket, m_param.addrun(), sizeof(sockaddr_un));    //本地套接字：m_param.addrun()，返回本地套接字地址，本地套接字类型sockaddr_un

            if (ret == -1) return -3;

            //监听套接字，已完成连接队列和未完成连接队列之和为 32
            ret = listen(m_socket, 32);

            if (ret == -1)return -4;
        }

        // 如果是非阻塞的，设置套接字为非阻塞，网络通信软件开发中，异步非阻塞套接字是用的最多的。平常所说的C/S（客户端/服务器）结构的软件就是异步非阻塞模式的。
        if (m_param.attr & SOCK_ISNONBLOCK) {

            int option = fcntl(m_socket, F_GETFL);      //获取套接字的标志
            //int fcntl (int __fd, int __cmd, ...);     int fcntl(int fd, int cmd ,struct flock* lock);
            // 获取失败
            if (option == -1)return -5;

            // 添加非阻塞标志
            option |= O_NONBLOCK;
            // 将改变后的标志，又重新设置到这个套接字上
            ret = fcntl(m_socket, F_SETFL, option);
            // 设置失败
            if (ret == -1)return -6;
        }

        //初始化完成，服务器初始化m_status才是0,初始化完成后，要置为1。客户端在前面的判断已经置为2了，所有不会执行这一步
        if(m_status == 0) m_status = 1;
        return 0;
    }
    //连接
    // 服务器 accept
    // 客户端 connect
    // 对于udp这里可以忽略
    virtual int Link(CSocketBase** pClient = NULL) {
        // 出错返回，未初始化或套接字创建失败
        if (m_status <= 0 || (m_socket == -1))return -1;

        int ret = 0;
        // 如果是服务器，就accept
        if (m_param.attr & SOCK_ISSERVER) {
            //服务端pclient不能为NULL
            if (pClient == NULL)return -2;

            CSockParam param;       //默认是客户端，阻塞，tcp
            int fd = -1;
            socklen_t len = 0;
            if (m_param.attr & SOCK_ISIP) {         // 网络套接字
                param.attr |= SOCK_ISIP;    // CSockParam 默认是本地，所以这里要修改
                len = sizeof(sockaddr_in);
                fd = accept(m_socket, param.addrin(), &len);
            }
            else {                                  //本地套接字
                len = sizeof(sockaddr_un);
                fd = accept(m_socket, param.addrun(), &len);
            }

            //`int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);`
            //- 从已完成连接队列里提取一个新的连接，
            //- 然后创建一个新的已连接套接字（通信套接字，cfd），
            //- 使用这个已连接套接字和当前连接的客户端通信
            // addr保存客户端地址的信息结构体

            //接收失败
            if (fd == -1)return -3;

            //如果接收成功，
            *pClient = new CSocket(fd);
            if (*pClient == NULL)return -4;
            // 初始化当前套接字cfd，由当前套接字和客户端通信
            ret = (*pClient)->Init(param);

            // 失败
            if (ret != 0) {
                delete (*pClient);
                *pClient = NULL;
                return -5;
            }
        }
        // 如果是客户端，之间connect
        else {
            if (m_param.attr & SOCK_ISIP)
                ret = connect(m_socket, m_param.addrin(), sizeof(sockaddr_in));
            else
                ret = connect(m_socket, m_param.addrun(), sizeof(sockaddr_un));
            if (ret != 0)return -6;
        }
        m_status = 2;
        return 0;
    }
    //发送数据
    virtual int Send(const Buffer& data) {

        // 必须要连接成功后，并且套接字已创建，才能发送数据
        if (m_status < 2 || (m_socket == -1))return -1;

        //index 用来记录已经写了多少内容
        ssize_t index = 0;
        while (index < (ssize_t)data.size()) {
            // 第二个参数为data剩下还要写的位置，第三个参数为data剩下还有多少要写
            ssize_t len = write(m_socket, (char*)data + index, data.size() - index);
            if (len == 0)return -2;
            if (len < 0)return -3;
            index += len;
        }
        return 0;
    }
    //接收数据 大于零，表示接收成功 小于 表示失败 等于0 表示没有收到数据，但没有错误
    virtual int Recv(Buffer& data) {

        if (m_status < 2 || (m_socket == -1))return -1;

        data.resize(1024*1024);
        ssize_t len = read(m_socket, data, data.size());

        //接收成功
        if (len > 0) {
            // 按照接收的内容长度作调整
            data.resize(len);
            return (int)len;//收到数据
        }
        data.clear();
        if (len < 0) {
            // 被中断EINTR
            if (errno == EINTR || (errno == EAGAIN)) {
                data.clear();
                return 0;//没有数据收到
            }
            return -2;//发送错误
        }
        return -3;//套接字被关闭
    }
    //关闭连接
    virtual int Close() {
        return CSocketBase::Close();
    }
};


#endif //EPLAYER_SOCKET_H
