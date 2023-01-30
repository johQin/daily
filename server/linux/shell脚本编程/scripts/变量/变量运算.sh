# 1 expr
echo "-----------------expr------------"
# 加
expr 1 + 3

num1=10
num2=20
expr $num1 + $num2

sum=`expr $num1 + $num2`
echo $sum

# 乘
expr $num1 \* $num2

# 2 $(())
echo '---------------$(())--------------'
echo $(($num1 * $num2))
echo $((num1 * num2))
echo $((5-3*2))
echo $(((3+2)*5))
echo $((2**10))
sum=$((num1 + num2))
echo $sum

# 3 $[]
echo '---------------$[]--------------'
echo $[2+2]
echo $[2**10]
num1=$[2**10]
echo $num1

# 4 let
echo '---------------let--------------'
let num3=1+2
echo $num3
let i++;echo $i