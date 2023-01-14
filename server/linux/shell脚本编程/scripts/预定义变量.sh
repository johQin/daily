# $0 命令的名字，带路径
echo "所有参数\$0：$0"
echo "所有参数\$*：$*"
echo "所有参数\$@：$@"
echo "参数个数\$#：$#"
echo "当前进程PID\$\$：$$"

# 检查命令是否携带参数
if [ $# -eq 0 ];then
    # 反引号是用来先执行命令，basename输出文件名
    echo "此命令: `basename $0` 需要携带参数"
    exit
fi

# -f用来检测$1所指的文件是不是文件
if [ ! -f $1 ];then
    # !取反，如果不是文件
    echo "error file! 位置参数1不是文件"
    exit
fi

# 循环ping ip
for ip in `cat $1`
do
    # echo $ip
    ping -c1 $ip &> /dev/null
    if [ $? -eq 0 ];then
        echo "${ip} is up."
    else
        echo "${ip} is down."
    fi
done

