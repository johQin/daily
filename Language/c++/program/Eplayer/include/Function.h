//
// Created by buntu on 2023/7/25.
//

#ifndef EPLAYER_FUNCTION_H
#define EPLAYER_FUNCTION_H
#include<functional>
#include<unistd.h>
#include<sys/types.h>       //类型的定义
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
#endif //EPLAYER_FUNCTION_H
