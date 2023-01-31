#!/usr/bin/bash
read -p 'please input username: ' user
if [ ! -z "$user" ]; then
    echo "input error"
    exit
fi

id $user &>/dev/null
if [ "$?" -ne 0 ];then
    echo "$user is unexited"
    exit
fi

read -p "are you sure? [y/n]" action

# if语句的写法
# if [ "$action" = "y" -o "$action" = "Y" -o "$action" = "yes" -o  "$action" = "YES" ]; then
#     userdel -r $user
#     echo "$user is deleted successfully"
# else
#     echo "action is canceled"
# fi

# case语句的写法
case $action in
"y" | "Y" | "yes" | "YES")
    userdel -r $user
    echo "$user is deleted successfully"
    ;;
*)
    echo "action is canceled"
    ;;
esac