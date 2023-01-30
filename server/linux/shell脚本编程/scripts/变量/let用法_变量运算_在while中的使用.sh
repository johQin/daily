#!/usr/bin/bash
ip=127.0.0.1
i=1
while [$i -le 5]
do
    ping -c1 $ip &>dev/null
    if [$? -eq 0];then
        echo "$ip is up..."
    fi
    let i++
done