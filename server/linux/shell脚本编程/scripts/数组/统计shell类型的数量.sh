#!/usr/bin/bash
declare -A shells

while read line
do
    type=`echo $line | awk -F ":" '{print $NF}'`
    let shells[$type]++
done </etc/passwd

for i in ${!shells[@]}
do
    echo "$i : ${shells[$i]}"
done