#!/usr/bin/bash
doublenum() {
    read -p 'please input a number： ' num
    # echo 'computing...'，注意此时函数里只能有一个echo,或者标准输出只能有一个
    echo $((2 * num))
}

result=`doublenum`
echo "doublenum return value: $result"