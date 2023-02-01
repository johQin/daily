# 1.函数定义，声明局部变量
fun1() {
	local a=100
}
# 调用
fun1
echo "a：$a"	# 无法获取

# 2.函数定义，声明全局变量
fun2() {
	b=100
}
fun2
echo "b：$b" # 100

# 3.函数定义，命令替换下的全局变量
fun3() {
	c=100
	echo 200
}
# 命令替代，相当于在子shell中执行，执行后，c就消失了
result=`fun3;echo "子shellc：$c"`
echo "c：$c" # 无法获取