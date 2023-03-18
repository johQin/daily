//��Ӧ�Ѽ򵥰�
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include "wrap.h"

#define _BUF_LEN_  1024
#define _EVENT_SIZE_ 1024

//全局的epoll树根
int gepfd = 0;

// 事件驱动的结构体
typedef struct xx_event{
    int fd;
    int events;
    void (*call_back)(int fd,int events,void *arg);
    void *arg;
    char buf[1024];
    int buflen;
    int epfd;
}xevent;

xevent myevents[_EVENT_SIZE_+1];

void readData(int fd,int events,void *arg);

//添加事件
//eventadd(lfd,EPOLLIN,initAccept,&myevents[_EVENT_SIZE_-1],&myevents[_EVENT_SIZE_-1]);
void eventadd(int fd,int events,void (*call_back)(int ,int ,void *),void *arg,xevent *ev)
{
    ev->fd = fd;
    ev->events = events;
    //ev->arg = arg;//代表结构体自己，通过arg得到结构体的所有信息
    ev->call_back = call_back;

    struct epoll_event epv;
    epv.events = events;
    epv.data.ptr = ev;//核心思想
    epoll_ctl(gepfd,EPOLL_CTL_ADD,fd,&epv);//上树
}

//修改事件
//eventset(fd,EPOLLOUT,senddata,arg,ev);
void eventset(int fd,int events,void (*call_back)(int ,int ,void *),void *arg,xevent *ev)
{
    ev->fd = fd;
    ev->events = events;
    //ev->arg = arg;
    ev->call_back = call_back;

    struct epoll_event epv;
    epv.events = events;
    epv.data.ptr = ev;
    epoll_ctl(gepfd,EPOLL_CTL_MOD,fd,&epv);//修改树节点
}

//删除事件
void eventdel(xevent *ev,int fd,int events)
{
	printf("begin call %s\n",__FUNCTION__);

    ev->fd = 0;
    ev->events = 0;
    ev->call_back = NULL;
    memset(ev->buf,0x00,sizeof(ev->buf));
    ev->buflen = 0;

    struct epoll_event epv;
    epv.data.ptr = NULL;
    epv.events = events;
    epoll_ctl(gepfd,EPOLL_CTL_DEL,fd,&epv);//下树
}

//发送数据
void senddata(int fd,int events,void *arg)
{
    printf("begin call %s\n",__FUNCTION__);

    xevent *ev = arg;
    Write(fd,ev->buf,ev->buflen);
    eventset(fd,EPOLLIN,readData,arg,ev);
}

//读数据
void readData(int fd,int events,void *arg)
{
    printf("begin call %s\n",__FUNCTION__);
    xevent *ev = arg;

    ev->buflen = Read(fd,ev->buf,sizeof(ev->buf));//把读到的数据存在buf中
    if(ev->buflen>0) //读到数据
	{	
		//void eventset(int fd,int events,void (*call_back)(int ,int ,void *),void *arg,xevent *ev)
        eventset(fd,EPOLLOUT,senddata,arg,ev);
        // 在这里，并没有直接的写数据，而修改节点的监听为写，
        // 写数据的时候，可能写缓存区已满，缓存区不是可写状态，
        // 当修改监听为写时，一旦写缓存区可写（水平触发），那么就可以发送数据了
        // 不过这里又要回到main的while里面，重新通过epoll_wait监听可写，然后执行对应的senddata
        // 因为存储收的数据在buf中，所以，在senddata中可以使用，读到的数据。
        // 而senddata，在发完数据之后，又要将节点的监听修改为听，等待下一次写数据
        // 如此循环往复，听-> 写 -> 听 ->写

    }
	else if(ev->buflen==0) //客户端连接断开
	{
        Close(fd);
        eventdel(ev,fd,EPOLLIN);
    }

}
//新连接请求处理
//lfd，EPOLLIN, ptr
void initAccept(int fd,int events,void *arg)
{
    printf("begin call %s,gepfd =%d\n",__FUNCTION__,gepfd);//__FUNCTION__ 预定义宏，函数名

    int i;
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    int cfd = Accept(fd,(struct sockaddr*)&addr,&len);//是否会阻塞？不会，因为有读事件来了。
	
	//查找myevents数组中可用的位置
    for(i = 0 ; i < _EVENT_SIZE_; i ++)
	{
        if(myevents[i].fd==0)
		{
            break;
        }
    }

    //添加读事件，上树

    //lfd的回调和cfd的回调不一样，lfd的initAccept，cfd的readData
    eventadd(cfd,EPOLLIN,readData,&myevents[i],&myevents[i]);
}

int main(int argc,char *argv[])
{
	//创建socket
    int lfd = Socket(AF_INET,SOCK_STREAM,0);

    //设置端口复用
    int opt = 1;
    setsockopt(lfd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));

	//绑定
    struct sockaddr_in servaddr;
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(8888);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    Bind(lfd,(struct sockaddr*)&servaddr,sizeof(servaddr));
    
	//监听
    Listen(lfd,128);

	//创建epoll树根节点
    gepfd = epoll_create(1024);
    printf("gepfd === %d\n",gepfd);

    struct epoll_event events[1024];

    //添加最初的事件，将监听的描述符lfd上树，放在数组的最后一个。//lfd的回调和cfd的回调不一样，lfd的initAccept，cfd的readData
    eventadd(lfd,EPOLLIN,initAccept,&myevents[_EVENT_SIZE_],&myevents[_EVENT_SIZE_]);
    //void eventadd(int fd,int events,void (*call_back)(int ,int ,void *),void *arg,xevent *ev)

    while(1)
	{
        int nready = epoll_wait(gepfd,events,1024,-1);
		if(nready<0) //调用epollwait失败
		{
			perr_exit("epoll_wait error");
			
		}
        else if(nready>0) //调用epoll_wait成功,返回有事件发生的描述符个数
		{
            int i = 0;
            for(i=0;i<nready; i++)
			{
                xevent *xe = events[i].data.ptr;//取ptr指向结构体的地址
                printf("fd=%d\n",xe->fd);

                // 监听的事件和返回的事件，位与一下
                if(xe->events & events[i].events)
				{
                    xe->call_back(xe->fd,xe->events,xe);//调用事件的对应回调
                }
            }
        }
    }

	//关闭监听文件描述符
	Close(lfd);

    return 0;
}

// 为什么要可读以后设置可写，然后一直交替？
// 答：服务器的基本工作无非数据的收发，epoll反应堆模型准从TCP模式，一问一答。服务器收到了数据，再给与回复，是目前绝大多数服务器的情况。

// 服务器能收到数据并不是一定能写数据
// 假设一 ：服务器接收到客户端数据，刚好此时客户端的接收滑动窗口满，我们假设不进行可写事件设置，并且客户端是有意让自己的接收滑动窗口满的情况(黑客)。那么，当前服务器将随客户端的状态一直阻塞在可写事件，除非你自己在写数据时设置非阻塞+错误处理
// 假设二 ：客户端在发送完数据后突然由于异常原因停止，这将导致一个FIN发送至服务器，如果服务器不设置可写事件监听，那么在接收数据后写入数据会引发异常SIGPIPE，最终服务器进程终止。