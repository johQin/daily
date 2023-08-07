//
// Created by buntu on 2023/8/3.
//
#include "Logger.h"
#include <stdarg.h>
// 本文件为LogInfo的几个构造器的实现
// 也为TRACE，LOG，DUMP系列宏的对应的构造器实现

// 因为CLoggerServer里用了LoggerInfo，而LoggerInfo里使用了CLoggerServer，所以出现了交叉引用
// 携带用户自定义的日志信息。TRACE系列宏的实现
LogInfo::LogInfo(
        const char* file, int line, const char* func,
        pid_t pid, pthread_t tid, int level,
        const char* fmt, ...
)
{

    const char sLevel[][8] = {
            "INFO","DEBUG","WARNING","ERROR","FATAL"
    };
    char* buf = NULL;
    bAuto = false;
    // 日志信息头
    int count = asprintf(&buf, "%s(%d):[%s][%s]<%d-%d>(%s) ",
                         file, line, sLevel[level],
                         (char*)CLoggerServer::GetTimeStr(), pid, tid, func);   //将格式化后的数据写入数据缓冲区buffer，当写入的长度超过buffer长度时，将为buffer自动分配内存，以存放数据。返回写入字符串的长度
    // 如果写入字符串大于0,
    if (count > 0) {
        m_buf = buf;
        free(buf);
    }
    else return;

    // 用户数据，可变参数的内容放的是用户数据
    // 可变参数的处理
    va_list ap;             //获取到第一个参数的指针
    va_start(ap, fmt);      //va_start的第二参数为last_arg，这个参数为可变参数前的最后一个固定参数，从上面的参数列表可以看到可变参数之前，最后固定一个参数为fmt
                            //通过这个最后一个固定参数可以获得，可变参数的起始位置
    count = vasprintf(&buf, fmt, ap);       //为可变参数创建一个格式化的字符串，并将其存储在动态分配的内存中。它的使用方法与 printf 类似，但它不会将结果打印到标准输出流中，而是将其存储在一个指向字符数组的指针中。

    if (count > 0) {        // 如果count小于0,则证明分配内存失败
        m_buf += buf;
        free(buf);      // 如果分配成功，必须要通过free来手动释放buf
    }
    va_end(ap);
}
// LOG系列宏
// 不携带用户自定义的日志，仅记录代码相关的执行流信息（也就是日志头信息）
LogInfo::LogInfo(const char* file, int line, const char* func, pid_t pid, pthread_t tid, int level)
{//自己主动发的 流式的日志

    // 日志头
    bAuto = true;
    const char sLevel[][8] = {
            "INFO","DEBUG","WARNING","ERROR","FATAL"
    };
    char* buf = NULL;
    int count = asprintf(&buf, "%s(%d):[%s][%s]<%d-%d>(%s) ",
                         file, line, sLevel[level],
                         (char*)CLoggerServer::GetTimeStr(), pid, tid, func);
    if (count > 0) {
        m_buf = buf;
        free(buf);
    }

}

//DUMP系列宏
LogInfo::LogInfo(
        const char* file, int line, const char* func,
        pid_t pid, pthread_t tid, int level,
        void* pData, size_t nSize
)
{
    // 日志头
    const char sLevel[][8] = {
            "INFO","DEBUG","WARNING","ERROR","FATAL"
    };
    char* buf = NULL;
    bAuto = false;
    int count = asprintf(&buf, "%s(%d):[%s][%s]<%d-%d>(%s)\n",
                         file, line, sLevel[level],
                         (char*)CLoggerServer::GetTimeStr(), pid, tid, func);
    if (count > 0) {
        m_buf = buf;
        free(buf);
    }
    else return;

    //
    Buffer out;
    size_t i = 0;
    char* Data = (char*)pData;
    for (; i < nSize; i++)
    {
        char buf[16] = "";
        // 和sprintf一样，不过有长度限制
        snprintf(buf, sizeof(buf), "%02X ", Data[i] & 0xFF);        // Data[i]是一个字符，转成整数会进行符号位扩展。所以为了不让它扩展，就直接与上0xFF
        m_buf += buf;
        // 每一行的内容形式为 00 01 F0 65……  ; a b c d……
        if (0 == ((i + 1) % 16)) {      // 每Datat 里 16 个字符一行，
            m_buf += "\t; ";
            for (size_t j = i - 15; j <= i; j++) {
                // 如果字符在ASCII码显示范围，则直接衔接
                if ((Data[j] & 0xFF) > 31 && ((Data[j] & 0xFF) < 0x7F)) {
                    m_buf += Data[i];
                }
                // 如果字符不在ASCII码显示范围，则用一个.代替
                else {
                    m_buf += '.';
                }
            }
            m_buf += "\n";
        }
    }
    //处理尾巴，最后没有满16个，要单独处理
    size_t k = i % 16;
    if (k != 0) {

        // 前面的一半的16进制数没有满16个，因此要用空格填充占位，以此来达到后面一半的原样字符可以和上面的内容对齐
        for (size_t j = 0; j < 16 - k; j++) m_buf += "   ";
        m_buf += "\t; ";
        // 处理后一半原样字符
        for (size_t j = i - k; j <= i; j++) {
            if ((Data[j] & 0xFF) > 31 && ((Data[j] & 0xFF) < 0x7F)) {
                m_buf += Data[i];
            }
            else {
                m_buf += '.';
            }
        }
        m_buf += "\n";
    }
}
// 这里的析构是什么意思，
LogInfo::~LogInfo()
{
    // 在logInfo析构的时候，并且bAuto为true的情况下，记录一个日志
    if (bAuto) {
        CLoggerServer::Trace(*this);
    }
}
