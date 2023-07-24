//
// Created by buntu on 2023/7/24.
//
#include <iostream>
#include<unistd.h>
#include<functional>
#include<memory.h>
#include<sys/socket.h>  //socket_pair
#include<sys/types.h>   //类型的定义
#include<sys/stat.h>    //system的状态值
#include<fcntl.h>       //文件操作
#include<signal.h>
#include "../../include/MultiProcess.h"
class CFunctionBase{
public:
    virtual ~CFunctionBase(){}
    virtual int operator()()=0;     //所有子类必须实现()符号重载
};

template<typename _FUNCTION_,typename... _ARGS_>
class CFunction:public CFunctionBase{
public:
    // 这里必须对m_binder进行初始化，否则会报错
    //  error: no matching function for call to ‘std::_Bind_result<int, int (*(CProcess*))(CProcess*)>::_Bind_result()’
    CFunction(_FUNCTION_ func,_ARGS_... args)
            :m_binder(std::forward<_FUNCTION_>(func),std::forward<_ARGS_>(args)...)         //std::forward实现完美转发
    {

    }
    virtual ~CFunction(){}
    virtual int operator()(){
        return m_binder();
    }
    // 这里必须加typename，告知编译器这里是一个类型。否则会报错
    //error: need ‘typename’ before ‘std::_Bindres_helper<int, _FUNCTION_, _ARGS_ ...>::type’ because ‘std::_Bindres_helper<int, _FUNCTION_, _ARGS_ ...>’ is a dependent scope
    // std::_Bindres_helper<int, _FUNCTION_,_ARGS_...>::type m_binder;
    typename std::_Bindres_helper<int, _FUNCTION_,_ARGS_...>::type m_binder;
};


class CProcess{
public:
    CProcess(){
        m_func = NULL;
        memset(pipes,0,sizeof(pipes));
    }
    virtual ~CProcess(){
        //因为有new，所以需要析构函数来释放
        if(m_func !=NULL){
            delete m_func;
            m_func = NULL;
        }

    }
    template<typename _FUNCTION_,typename... _ARGS_>        // 可变参数模板
    int SetEntryFunction(_FUNCTION_ func,_ARGS_... args){
        m_func = new CFunction<_FUNCTION_,_ARGS_...>(func,args...);       //“...” 定义的时候在类型后加，使用的时候，在实参后加
        return 0;
    }
    int CreateSubProcess(){
        if(m_func == NULL) return -1;
        int ret =socketpair(AF_LOCAL,SOCK_STREAM,0,pipes);

        if(ret == -1) return -2;
        printf("pipes 生成成功\n");
        pid_t pid = fork();         // 子进程复制了父进程的所有内容，包括这里的pipes，所以父进程有一个pipes[2],子进程也有一个pipes[2],这两个pipes通过socketpair函数连接在一起。
        if (pid==-1) return -3;
        printf("fork 成功，pid：%d\n",pid);
        // 子进程
        if(pid == 0){
            printf("sub process (%d):<%s> pid=%d\n", __LINE__, __FUNCTION__, getpid());
            close(pipes[1]);        //关闭写端
            pipes[1] = 0;
            (*m_func)();
            exit(0);
        }

        // 父进程
        printf("父进程\n");
        close(pipes[0]);            //关闭读端
        pipes[0] = 0;

        m_pid = pid;
        return 0;
    }

    // 主进程发送文件描述符
    int SendFD(int fd){

        // 这里的消息主体是不重要的
        struct msghdr msg {};
        iovec iov[2];
        char buf[][10] ={"qq","kk"};
        iov[0].iov_base = buf[0];         // 最后我们传递的是文件描述符，所以这里的消息缓冲不重要，作为一种冗余放在这里。
        iov[0].iov_len = sizeof(buf[0]);
        iov[1].iov_base = buf[1];
        iov[1].iov_len = sizeof(buf[1]);
        msg.msg_iov= iov;
        msg.msg_iovlen = 2;


        // 下面的数据才是我们需要传递的数据，之前的都是幌子
        cmsghdr * cmsg =  new (std::nothrow) cmsghdr;       //std::nothrow参数来告诉new操作符不要抛出异常，而是返回一个空指针。然后，我们检查指针p是否为空，以确定内存是否分配成功。
        if(cmsg == nullptr) return -1;
        bzero(cmsg,sizeof(cmsghdr));        //前两句可以简写为c的方式：cmsghdr * cmsg = (cmsghdr *) calloc(1, CMSG_LEN(sizeof(int)));如果采用这种，那么下面的delete需要修改为free
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));     // #define CMSG_LEN(len)   (CMSG_ALIGN (sizeof (struct cmsghdr)) + (len))
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        *(int*) CMSG_DATA(cmsg) = fd;


        msg.msg_control = cmsg;     // control msg
        msg.msg_controllen = cmsg->cmsg_len;

        ssize_t ret = sendmsg(pipes[1],&msg,0);
        delete cmsg;
        if(ret ==-1)return -2;

        return 0;
    }

    int RecvFD(int & fd){
        printf("子进程%s\n",__FUNCTION__);
        struct msghdr msg{};
        iovec iov[2];
        char buf[][10] = {"",""};
        iov[0].iov_base = buf[0];
        iov[0].iov_len = sizeof(buf[0]);
        iov[1].iov_base = buf[1];
        iov[1].iov_len = sizeof(buf[1]);
        msg.msg_iov = iov;
        msg.msg_iovlen = 2;

        cmsghdr * cmsg = (cmsghdr *) calloc(1, CMSG_LEN(sizeof(int)));
        if (cmsg ==NULL) return -1;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;

        msg.msg_control = cmsg;
        msg.msg_controllen = cmsg->cmsg_len;

        ssize_t ret = recvmsg(pipes[0],&msg,0);
        if (ret ==-1) {
            free(cmsg);
            return -2;
        }
        fd = *(int *) CMSG_DATA(cmsg);
        return 0;
    }
    static int SwitchDaemon(){
        //主进程
        pid_t ret = fork();
        if(ret<0) return -1;
        if(ret>0) exit(0);      //主进程退出

        // 父进程继续
        // 2.创建新会话
        ret =setsid();
        if(ret < 0) return -2;          // setsid error
        ret = fork();
        if(ret < 0) return -3;
        if(ret > 0) exit(0);    //父进程退出

        //孙进程如下，
        //3.设置工作目录
        chdir("/tmp");
        //4.重设文件掩码
        umask(0);
        //5.关闭从父进程继承下来的文件描述符
        for(int i=0;i<getdtablesize();i++) close(i);
        // signal(SIGCHLD, SIG_IGN);                // 这句代码存疑，一般signal(SIGCHLD, SIG_IGN)放在父进程中，以处理僵尸进程的情况。可这里放在了孙子进程中，有点奇怪
        // 守护进程的执行内容
        // ....
        //
        //6.执行任务(每5秒记录一次系统时间)
        //    while(1)
        //    {
        //        system("echo `date` >> time.txt");
        //        sleep(5);
        //    }
        return 0;
    }

private:
    // 这里为什么不直接用CFunction，而要用父类，就是为了避免模板的传染性，这里通过基类隔离一下，当前类（CProcess）就不会被传染
    // 如果这里用了CFunction<T1,T2>, 由于CFunction是一个模板类，那么在声明其类型的时候肯定使用类型模板T1，这就会导致CProcess变成一个模板类。
    CFunctionBase* m_func;
    pid_t m_pid;
    int pipes[2];       //0读，1写。子进程关闭写端，父进程关闭读端。
};
int CreateLogServer(CProcess * proc){
    printf("pid=%d <%s> \n", getpid(), __FUNCTION__);
    return 0;
}
int CreateClientServer(CProcess * proc){
    printf("子进程执行函数：CreateClientServer\n");
//    printf("%s(%d):<%s> pid=%d\n", __FILE__, __LINE__, __FUNCTION__, getpid());
    int fd = -1;

    sleep(1);
    int ret = proc->RecvFD(fd);
    printf("(%d):<%s>%d,%d\n",__LINE__, __FUNCTION__,ret, fd);
    printf("子进程fd：%d\n",fd);
    char buf[10] = "";
    lseek(fd, 0, SEEK_SET);
    read(fd, buf, sizeof(buf));
    printf("读到的内容：%s",buf);
    close(fd);
    return 0;
}
int testMultiProcess(){
    // 开启守护进程模式
    //CProcess::SwitchDaemon();
    CProcess procclient;
    //proclog.SetEntryFunction(CreateLogServer,&proclog);
    procclient.SetEntryFunction(CreateClientServer,&procclient);    // 可以自动推导类型
    procclient.CreateSubProcess();

    int fd = open("./test.txt",O_RDWR | O_CREAT | O_APPEND);
    if (fd ==-1) return -3;
    write(fd,"qqqkkkhhh",9);
    int ret =procclient.SendFD(fd);
    if (ret != 0) printf("errno:%d msg:%s\n", errno, strerror(errno));
    close(fd);
    printf("父进程\n");
    return 0;
}