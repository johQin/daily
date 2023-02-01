thread_num=5
tmpfifo='/tmp/$$.fifo'

mkfifo $tempfifo    # 创建一个fifo管道文件
exec 8<> $tempfifo  # 指定描述符序号为8
rm $tempfifo    # 删除文件不会影响文件描述符

for j in `seq $thread_num`
do
    echo >&8    
    # echo 每次写入一个0a字符（换行符，不是没有内容哈），给文件描述符8，总共写了5次，文件里面有5个字符
    # 还有这里的重定向符“>（覆盖）”，为什么不是">>(追加)"，因为管道文件是无法覆盖的，写一个就是一个，是无法覆盖的
done
for i in `seq 100`
do
    read -u 8
    # -u 指定文件描述符 
    # read必须读到内容，否则将会停滞在这里，而管道文件的内容读一个字符，少一个字符，所以每次最多只能有5个进程
    {
        ip="10.80.5.${i}"
        ping -c1 -w1 $ip
        if [ $? -eq 0 ];then
            echo "$user is up"
        else
            echo "$user is down"
        fi
        echo >&8 # 每次进程结束，需要往管道文件里，加一个0a字符，相当于释放一个进程
    }&
done
wait
# 释放文件描述符
exec 8>&-
echo "all finished"