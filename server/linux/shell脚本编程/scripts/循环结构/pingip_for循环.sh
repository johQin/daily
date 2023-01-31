#!/usr/bin/bash
# 这里有两种方式创建一个序列，一个是{start..end}，一个是seq start end
for i in {2..254}
# for i in `seq 2 254`
do
    {
        ip="10.80.5.$i"
        ping -c1 -W1 $ip &>/dev/null
        if [ $? -eq 0 ]; then
            echo "$ip" | tee -a ip_up.txt
            echo "is up"
        else
            echo "$ip is down"
        fi
    }& # 将花括号里面的命令丢到后台执行
done
wait # 等待所有后台进程结束再执行后面的的命令
echo "finished"