#!/usr/bin/bash
# sex.txt
# qqq f
# kkk m
# hhh x
# rrr f
# yyy f
# xxx m
# ttt x

declare -A sex

while read line
do
    # 把需要统计的对象作为数组的索引
    type=`echo $line | awk '{print:$2}'`
    # 这里可以在关联数组里面主动加键名
    let sex[$type]++
done < sex.txt

for i in ${!sex[@]}
do
    echo "$i : ${sex[$i]}"
done