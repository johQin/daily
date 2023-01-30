#!/usr/bin/bash
# 输入创建的用户数
read -p "Please input number" num
# 输入必须是数字
if [[ ! "$num" =~ ^[0-9]+$ || "$num" =~ ^0+$ ]]; then
    echo "error number"
    exit
fi

# 输入用户名的前缀
read -p "Please input prefix name" prefix
# 输入非空用户名前缀
if [ -z "$prefix" ]; then #这里prefix一定要使用双引号括起来，因为prefix未定义的情况下，会报语法错误
    echo "error prefix"
    exit
fi
for i in "seq $num"
do
    user=$prefix$i
    useradd $user
    echo "123456" | passwd --stdin $user &> /dev/null
    if [ $? -eq 0 ];then
        echo "$user is created successfully"
    fi
done