#!/usr/bin/bash
old_IFS=$IFS
IFS=$'\n'
for line in `cat /etc/hosts`
do
    hosts[i++]=$line
done
IFS=$old_IFS

for i in ${!hosts[@]}
do
    echo "${i} : ${hosts[$i]}"
done