//
// Created by buntu on 2023/7/25.
//

#ifndef EPLAYER_EPOLL_H
#define EPLAYER_EPOLL_H
#include<unistd.h>
#include<sys/epoll.h>
#include <vector>
#include <errno.h>
#include <sys/signal.h>
#include <memory.h>
#define EVENT_SIZE 128
class EpollData{
public:
    EpollData() { m_data.u64 = 0;}
    EpollData(void* ptr) { m_data.ptr = ptr; }
    explicit EpollData(int fd) { m_data.fd = fd;}
    explicit EpollData(uint32_t u32) {m_data.u32 = u32; }
    explicit EpollData(uint64_t u64) {m_data.u64 = u64; }
    EpollData(const EpollData& data) {m_data.u64 = data.m_data.u64; }   //赋值构造，因为是联合体，u64是最长的一段内存，所以一旦它被赋值，那么就整个被赋值了
public:
    EpollData& operator=(const EpollData& data){
        if (this != &data) m_data.u64 = data.m_data.u64;    // 如果不是自己赋值给自己，就需要手动赋值
        return *this;
    }
    EpollData& operator=(void* data) {
        m_data.ptr = data;
        return *this;
    }
    EpollData& operator=(int data) {
        m_data.fd = data;
        return *this;
    }
    EpollData& operator=(uint32_t data) {
        m_data.u32 = data;
        return * this;
    }
    EpollData& operator=(uint64_t data) {
        m_data.u64 = data;
        return * this;
    }
    //类型转换函数，可以将EpollData隐式转换为epoll_data_t，或epoll_data_t*类型
    operator epoll_data_t(){return m_data;}
    operator epoll_data_t() const {return m_data;}
    operator epoll_data_t* () {return &m_data;}
    operator const epoll_data_t*() const {return &m_data;}

private:
    epoll_data_t m_data;
    //联合体
//    typedef union epoll_data
//    {
//        void *ptr;
//        int fd;
//        uint32_t u32;
//        uint64_t u64;
//    } epoll_data_t;
};

using EPEvents = std::vector<epoll_event>;

class CEpoll{
public:
    CEpoll(){
        m_epoll = -1;
    };
    ~CEpoll(){
        Close();
    };

public:
    // 删除拷贝构造函数，子类不能重写
    CEpoll(const CEpoll&)=delete;
    // 删除重载的赋值运算符，子类不能重写
    CEpoll& operator=(const CEpoll&) = delete;
public:
    // 类型转换函数，可以将CEpoll隐式转换为int类型。   类似于运算符重载的写法，1、要求函数名称为要转换的类型 2、函数无形参
    operator int() const {return m_epoll;}
public:
    int Create(unsigned count){
        if (m_epoll != -1)return -1;
        m_epoll = epoll_create(count);
        if (m_epoll == -1)return -2;
        return 0;
    };
    ssize_t WaitEvents(EPEvents& events, int timeout = 10){
        if(m_epoll == -1) return -1;
        EPEvents evs(EVENT_SIZE);
        int ret = epoll_wait(m_epoll,evs.data(), (int)evs.size(), timeout);
        if (ret == -1) {
            if((errno == EINTR) || (errno == EAGAIN)){
                return 0;
            }
            return -2;
        }
        if(ret> (int) events.size()) {
            events.resize(ret);
        }
        memcpy(events.data(),evs.data(),sizeof(epoll_event) * ret);
        return ret;
    };
    int Add(int fd, const EpollData& data = EpollData((void *) 0),uint32_t events = EPOLLIN){
        if (m_epoll == -1)return -1;
        epoll_event ev = { events,data };
        int ret = epoll_ctl(m_epoll,EPOLL_CTL_ADD,fd,&ev);
        if (ret == -1)return -2;
        return 0;
    };
    int Modify(int fd, uint32_t events,const EpollData& data = EpollData((void*) 0)){
        if (m_epoll == -1)return -1;
        epoll_event ev = { events,data };
        int ret = epoll_ctl(m_epoll,EPOLL_CTL_MOD, fd, &ev);
        if (ret == -1)return -2;
        return 0;
    };
    int Del(int fd){
        if (m_epoll == -1)return -1;
        int ret = epoll_ctl(m_epoll,EPOLL_CTL_DEL, fd, NULL);
        if (ret == -1)return -2;
        return 0;
    };
    void Close(){
        if (m_epoll != -1) {
            int fd = m_epoll;
            m_epoll = -1;
            close(fd);
        }
    };
private:
    int m_epoll;
};
#endif //EPLAYER_EPOLL_H
