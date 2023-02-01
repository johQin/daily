for i in {2..254}
do
    {
        ip="10.80.5.$i"
        ping -c1 -w1 $ip &>/dev/null
        if [ $? -eq 0 ];then
            echo "$ip is up."
        fi
    }&
done
wait
echo "all finished"

j=2
while [ $j -le 254 ]
do
    {
        ip="10.80.5.$j"
        ping -c1 -w1 $ip &>/dev/null
        if [ $? -eq 0 ];then
            echo "$ip is up."
        fi
    }&
    let i++ # 这里不能扔到异步（后台）去自加
done
wait
echo "all finished"

k=2
util [ $k -gt 254 ]
do
    {
        ip="10.80.5.$k"
        ping -c1 -w1 $ip &>/dev/null
        if [ $? -eq 0 ];then
            echo "$ip is up."
        fi
    }&
    let k++
done
wait
echo "all finished"