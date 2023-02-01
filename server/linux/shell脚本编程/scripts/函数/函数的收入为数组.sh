#!/usr/bin/bash
num1=(1 2 3)
# 以空格做分隔
echo ${num1[@]}
arr_fun() {
    echo $*
    # 如果函数里的变量
    # 不加local，那么函数外也可以获取到此变量的值，该变量属于全局变量
    # 加local，那么函数外不可获取此变量的值，该变量属于局部变量
    local res=1
    
    # 如果用数组元素作为实参，那么函数体内通过$*,$@接收
    for i in $*
    # for i in $@
    do
        res=$[res * i]
    done
    echo $res
}
# 将数组的所有元素都做实参数
arr_fun ${num1[@]}
