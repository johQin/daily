#/bin/env bash
# 这样的shebang，可以自动去查找bash的位置

levelmultiple() {
    res=1
    # 函数里的$1是函数的位置参数，$1是函数的第一个参数
    for ((i=1;i<=$1;i++)) #类c写法
    do
        # res=$((res * i))
        let res*=i
    done
    echo "$1 的阶乘：$res"
}

# 最外层的$1是脚本的位置参数，$1是脚本的第一个位置参数
levelmultiple $1