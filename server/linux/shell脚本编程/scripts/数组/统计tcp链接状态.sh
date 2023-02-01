#!/usr/bin/bash
declare -A status
type=`ss -an | grep :80 | awk '{print $2}'`

for i in $type
do
    let status[$i]++
    let shells[$type]++
done

for j in ${!status[@]}
do
    echo "$j : ${status[$j]}"
done