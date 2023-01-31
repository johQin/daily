#!/usr/bin/bash
# 通过文件，批量创建用户，用户名和密码都在文件中

# 判断参数个数
if [ $# -eq 0 ];then
    echo "must input file parameters"
fi

# 判断文件是否存在
if [ ! -f $1 ]; then
    echo "file：$1 is unexisted"
fi

# 我们希望文件内容按回车分割，而不是按空格或tab分割
# 这里需要重新设置IFS，使用完毕后需要还原
IFSTemp=$IFS
IFS=$'\n'
# user.txt 内容
# abc 145566
# ttt 123456
# dfd 145987
for line in `cat user.txt`
do
    # 如果是空行，跳过
    if [ ${#line} -eq 0 ]; then
        continue
    fi
    # 分割每行内容
    user=`echo "$line" | awk '{print $1}'`
    pass=`echo "$line" | awk '{print $2}'`
    id $user &>/dev/null
    if [ $? -eq 0 ]; then
        echo "$user already exists"
    else
        useradd $user
        echo "$pass" | passwd --stdin $user &>/dev/null
        if [ $? -eq 0 ]; then
            echo "$user created successfully"
        fi
    fi
done
IFS=$IFSTemp