#include <iostream>
#include<unistd.h>
#include<functional>
class CFunctionBase{
public:
    virtual ~CFunctionBase(){}
    virtual int operator()()=0;     //所有子类必须实现()符号重载
};

template<typename _FUNCTION_,typename... _ARGS_>
class CFunction:public CFunctionBase{
public:
    CFunction(_FUNCTION_ func,_ARGS_... args){}
    virtual ~CFunction(){}
    virtual int operator()(){
        return m_binder();
    }
    std::_Bindres_helper<int, _FUNCTION_,_ARGS_...>::type m_binder;
};

class CProcess{
public:
    CProcess(){
        m_func = NULL;
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
        m_func = new CFunction(func,args...);       //定义的时候在类型后加，使用的时候，在实参后加
        return 0;
    }
    int CreateSubProcess(){
        if(m_func == NULL) return -1;
        pid_t pid = fork();
        if (pid==-1) return -2;

        // 子进程
        if(pid == 0){
            return (*m_func)();
        }
        // 父进程
        m_pid = pid;
        return 0;
    }
private:
    // 这里为什么不直接用CFunction，而要用父类，就是为了避免模板的传染性，这里通过基类隔离一下，当前类（CProcess）就不会被传染
    // 如果这里用了CFunction<T1,T2>, 由于CFunction是一个模板类，那么在声明其类型的时候肯定使用类型模板T1，这就会导致CProcess变成一个模板类。
    CFunctionBase* m_func;
    pid_t m_pid;
};
int CreaeLogServer(CProcess * proc){
    return 0;
}
int CreaeClientServer(CProcess * proc){
    return 0;
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    CProcess proclog,procclient;
//    proclog.set
    return 0;
}
