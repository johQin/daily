num1=(1 2 3)
arr_fun() {
    # $*本来就是参数的空格相隔的数组元素，用括号括起来就是定义数组
    local newarray=($*)
    local i
    for((i=0;i<$#;i++))
    do
        newarray[$i]=$(( ${newarray[$i]} * 5 ))
    done
    echo ${newarray[@]}
}
# arr_fun ${num1[@]}
result=`arr_fun ${num1[@]}`
echo "result: ${result[@]}"