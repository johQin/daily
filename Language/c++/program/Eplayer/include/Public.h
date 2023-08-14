//
// Created by buntu on 2023/8/14.
//

#ifndef EPLAYER_PUBLIC_H
#define EPLAYER_PUBLIC_H
#include <string>
#include <string.h>
// 继承字符串类
class Buffer :public std::string
{
public:
    // 提供一个默认构造函数
    Buffer() :std::string() {}
    // 按照空间大小构造缓冲区。构造Buffer的时候，会触发父类string的构造函数，并且调用resize函数调整空间大小
    Buffer(size_t size) :std::string() { resize(size); }
    Buffer(const std::string& str):std::string(str){}
    Buffer(const char * str):std::string(str){}
    Buffer(const char* str,size_t length)
            :std::string() {
        resize(length);
        memcpy((char*)c_str(), str, length);
    }
    Buffer(const char* begin, const char* end) :std::string() {
        long int len = end - begin;
        if (len > 0) {
            resize(len);
            memcpy((char*)c_str(), begin, len);
        }
    }
    // 继承std::string 转字符常量指针的用法，这就是为什么要继承的原因
    // 可以转换Buffer 为char * 指针
    operator char* () { return (char*)c_str(); }
    // 可以转换const Buffer 为 char * 指针
    operator char* () const { return (char*)c_str(); }
    // 可以转换const Buffer 为 const char * 指针
    operator const char* () const { return c_str(); }
};
#endif //EPLAYER_PUBLIC_H
