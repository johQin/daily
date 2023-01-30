read -p "请输入用户名： " user
if id $user &>/dev/null; then
    echo "用户已存在"
else
    useradd $user
    if [ $? -eq 0]; then
        echo "$user 已创建成功"
    fi
fi
