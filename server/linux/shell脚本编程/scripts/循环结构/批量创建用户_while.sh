while read line
do
    # 不需要做空行判断，因为read 空行会得到1的返回值
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
done < user.txt