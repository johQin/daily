#include <cstdio>
#include<unistd.h>       //fork
#include<functional>    //_Bindres_helper
#include<memory.h>      //memset
#include<sys/socket.h>  //socketpair
#include<sys/types.h>

class CFunctionBase {
public:
    virtual ~CFunctionBase(){}
    virtual int operator()() = 0;   //强迫所有子类必须重载“()”运算符
};

template<typename _FUNCTION_, typename... _ARGS_>
class CFunction {
public:
    CFunction(_FUNCTION_ func, _ARGS_... args) {

    }

    virtual ~CFunction(){}      //这个析构函数好像什么也没干，但它会触发成员变量的析构,也就是m_binder的析构

    std::_Bindres_helper<int, _FUNCTION_, _ARGS_...>::type m_binder;

    virtual int operator()() {
        return m_binder();
    }
};

class CProcess {
private:
    CFunctionBase* m_func;
    pid_t m_pid;            //子进程id
    int pipes[2];           //用于父进程传递fd给子进程,pipes[0]:读端，pipes[1]:写端
public:
    CProcess() {
        m_func = NULL;
        memset(pipes, 0, sizeof(pipes));
    }

    ~CProcess() {
        if (m_func != NULL) {
            delete m_func;
            m_func = NULL;
        }
    }

    template<typename _FUNCTION_,typename... _ARGS_>
    int SetProcessEntryFunction(_FUNCTION_ func, _ARGS_... args ) {
        m_func = new CFunction(func, args...);
        return 0;
    }

    int CreateSubProcess() {
        if (m_func == NULL) return -1;

        int spres = socketpair(AF_LOCAL, SOCK_STREAM, 0, pipes);
        if (spres == -1) return -2;

        pid_t pid = fork();
        //一次调用，两次返回，父子进程共享代码段。这就是为什么后面if else都会进入的原因
        //fork()成功时，在父进程中返回子进程的ID，在子进程中返回0

        if (pid == -1) return -3;   //子进程创建失败
        if (pid == 0) {             //子进程
            close(pipes[1]);        //关闭掉写,父进程单向传递给子进程，如果要双向传递需要建立两个socketpair
            pipes[1] = 0;
            return (*m_func)();     //这里会对_Function_(...args)进行一个真正调用
        }
        //主进程
        close(pipes[0]);            //关闭读端
        pipes[0] = 0;
        m_pid = pid;
        return 0;
    }

};

// 这两个函数还会封装的，不能面向过程
int CreateLogServer(CProcess* proc) {
    return 0;
}

int CreateClientHandleServer(CProcess* proc) {
    return 0;
}

int main()
{
    printf("%s 向你问好!\n", "EPlayerServer");
    CProcess procLog, procClientHandle;
    procLog.SetProcessEntryFunction(CreateLogServer, &procLog);
    int pret = procLog.CreateSubProcess();

    procClientHandle.SetProcessEntryFunction(CreateClientHandleServer, &procClientHandle);
    int cret = procClientHandle.CreateSubProcess();

    return 0;
}