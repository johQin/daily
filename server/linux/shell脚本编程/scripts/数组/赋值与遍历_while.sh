#!/usr/bin/bash
while read line
do
    hosts[i++]=$line
done < /etc/hosts

echo "hosts first: ${hosts{0}}"
echo

# ${!hosts[@]}返回数组的所有索引，以空格分割
for i in ${!hosts[@]}
do
    echo "${i} : ${hosts[$i]}"
done
