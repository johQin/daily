//
// Created by buntu on 2023/7/27.
//

#ifndef EPLAYER_THREAD_H
#define EPLAYER_THREAD_H
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <signal.h>
#include <map>
#include "Function.h"
#include<stdio.h>
#include<errno.h>

class CThread
{
public:
    CThread() {
        m_function = NULL;    // 线程的任务函数
        m_thread = 0;       //线程id
        m_bpaused = false;  //true 表示暂停 false表示运行中
    }

    template<typename _FUNCTION_, typename... _ARGS_>
    CThread(_FUNCTION_ func, _ARGS_... args)
        :m_function(new CFunction<_FUNCTION_, _ARGS_...>(func, args...))
        // m_function这里使用了初始化列表的形式来进行初始化。
        // 而括号里面可以是一个初始值（直接赋值给成员变量）
        // 或参数列表（他可以调用对应的构造函数）
    {
        m_thread = 0;
        m_bpaused = false;
    }

    ~CThread() {
        if(m_function != NULL){
            delete m_function;
            m_function = NULL;
        }
    }
public:
    CThread(const CThread&) = delete;
    CThread operator=(const CThread&) = delete;
public:
    // 设定线程的函数
    template<typename _FUNCTION_, typename... _ARGS_>
    int SetThreadFunc(_FUNCTION_ func, _ARGS_... args)
    {
        m_function = new CFunction<_FUNCTION_, _ARGS_...>(func, args...);
        if (m_function == NULL)return -1;
        return 0;
    }

    int Start() {
        pthread_attr_t attr;
                    //        typedef struct
                    //        {
                    //            int                       detachstate;   // 线程的分离状态
                    //            int                       schedpolicy;   // 线程调度策略
                    //            struct sched_param        schedparam;    // 线程的调度参数
                    //            int                       inheritsched;  // 线程的继承性
                    //            int                       scope;         // 线程的作用域
                    //            size_t                    guardsize;     // 线程栈末尾的警戒缓冲区大小
                    //            int                       stackaddr_set; // 线程的栈设置
                    //            void*                     stackaddr;     // 线程栈的位置
                    //            size_t                    stacksize;     // 线程栈的大小
                    //        } pthread_attr_t;
        int ret = 0;

        // 初始化线程属性，有初始化就必须要pthread_attr_destroy
        ret = pthread_attr_init(&attr);
        if (ret != 0)return -1;

        // 将线程设置为detach，就是线程执行结束后，自动释放资源（不需要主线程区pthread_join）
        ret = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        if (ret != 0)return -2;

        // 将线程限定在当前进程中，不与其他进程的线程争抢资源
        ret = pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);      //可以不设置，这是默认值
        if (ret != 0)return -3;

        // ThreadEntry为任务函数的入口，之所以这里不直接写任务函数，是因为想在任务函数里直接使用this，以方便获取相关的成员对象
        ret = pthread_create(&m_thread, &attr, &CThread::ThreadEntry, this);
                        //        int pthread_create(
                        //                pthread_t *thread, // 线程ID
                        //                const pthread_attr_t *attr, //线程自身的属性，可设置线程的状态等属性
                        //                void *(*start_routine) (void *), //交付线程执行的任务函数
                        //                void *arg// 传递给任务函数的参数，如果有多个参数，封装为一个结构体即可
                        //        );
        if (ret != 0)return -4;

        //
        m_mapThread[m_thread] = this;

        // 释放线程属性
        ret = pthread_attr_destroy(&attr);
        if (ret != 0)return -5;

        return 0;
    }
    // 暂停
    int Pause() {
        // 如果线程没有创建，那么就直接返回
        if (m_thread == 0)return -1;

        // 如果线程已暂停，那么就让他运行，只需要将标志位修改？不去做额外的动作？
        if (m_bpaused) {    //true 表示暂停 false表示运行中
            m_bpaused = false;
            return 0;
        }

        // 如果线程运行中
        m_bpaused = true;

        //SIGUSR1：由用户自定义的信号，向线程发信号，而不是kill线程，线程内如果实现了对应信号的handler，那么就去处理，这里的handler在ThreadEntry里设置了
        int ret = pthread_kill(m_thread, SIGUSR1);

        if (ret != 0) {
            m_bpaused = false;
            return -2;
        }
        return 0;
    }
    // 停止
    int Stop() {
        // 线程已创建成功
        if (m_thread != 0) {
            // 快速锁住if入口
            pthread_t thread = m_thread;
            m_thread = 0;

            // 设置在停止前等待的时间，如果为NULL，那么下面的pthread_timedjoin_np和pthread_join一样，一直等，等到线程自己结束
            timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 100 * 1000000;//100ms

            // 阻塞式，等待ts时间，看线程是否在等待时间内结束，如果超过等待时间还没结束，则返回ETIMEDOUT，不回收线程资源
            int ret = pthread_timedjoin_np(thread, NULL, &ts);
            if (ret == ETIMEDOUT) {
                pthread_detach(thread);     //设置线程在执行结束后，主动释放资源
                pthread_kill(thread, SIGUSR2);      //SIGUSR2：由用户自定义的信号，向线程发信号，而不是kill线程，线程内如果实现了对应信号的handler，那么就去处理，这里的handler在ThreadEntry里设置了
            }
        }
        return 0;
    }
    bool isValid()const { return m_thread == 0; }
private:
    //__stdcall（标准调用，不用传this指针，因为这是一个静态函数），任务函数的入口
    static void* ThreadEntry(void* arg) {
        // arg是pthread_create传给任务函数入口的参数，在Start被给为this，所以这里的arg就是this指针
        CThread* thiz = (CThread*)arg;

        // 注册线程内信号处理函数，用于处理pthread_kill发来的信号
        struct sigaction act = { 0 };
        sigemptyset(&act.sa_mask);  // 将监听的信号集合先初始为空
        act.sa_flags = SA_SIGINFO;          // sa_flags：特殊标志位，SA_SIGINFO：设定后面的信号处理函数为三参数
        act.sa_sigaction = &CThread::Sigaction;     // 绑定信号处理函数

        //绑定信号
        sigaction(SIGUSR1, &act, NULL);     // 来自于Pause函数的信号
        sigaction(SIGUSR2, &act, NULL);     // 来自于Stop函数的信号

        // 真正的调用，做任务的函数
        thiz->EnterThread();


        if (thiz->m_thread)thiz->m_thread = 0;
        pthread_t thread = pthread_self();//不是冗余，有可能被stop函数把m_thread给清零了
        auto it = m_mapThread.find(thread);
        if (it != m_mapThread.end())
            m_mapThread[thread] = NULL;
        pthread_detach(thread);
        pthread_exit(NULL);
    }

    //__thiscall（对象调用，需要this调用，调用实际的任务函数
    void EnterThread() {
        if (m_function != NULL) {
            int ret = (*m_function)();
            if (ret != 0) {
                printf("%s(%d):[%s]ret = %d\n", __FILE__, __LINE__, __FUNCTION__);
            }
        }
    }

    // 信号处理函数
    static void Sigaction(int signo, siginfo_t* info, void* context)
    {
        // 如果线程Pause
        if (signo == SIGUSR1) {
            pthread_t thread = pthread_self();
            auto it = m_mapThread.find(thread);
            if (it != m_mapThread.end()) {
                if (it->second) {
                    while (it->second->m_bpaused) {
                        if (it->second->m_thread == 0) {
                            pthread_exit(NULL);
                        }
                        usleep(1000);//1ms
                    }
                }
            }
        }
        // 如果线程Stop
        else if (signo == SIGUSR2) {//线程退出
            pthread_exit(NULL);
        }
    }

private:
    CFunctionBase* m_function;
    pthread_t m_thread;
    bool m_bpaused;     //true 表示暂停 false表示运行中
    static std::map<pthread_t, CThread*> m_mapThread;
};
#endif //EPLAYER_THREAD_H
