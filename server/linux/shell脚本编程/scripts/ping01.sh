#!/usr/bin/bash
ping -c1 www.baidu.com &> /dev/null && echo "baidu connected" || echo "baidu disconnected" 
echo $?

# #!/usr/bin/bash， #与!之间没有空格，这个叫shebang机制，程序的第一行，也仅第一行。声明这个脚本程序默认用哪个解释器执行，
# 如果通过 bash ./ping01.sh 执行，则就显式通过 bash 这个解释器执行，如果通过 ./ping01.sh 执行，则shebang机制就会生效，默认通过 #! 声明的解释器执行
# 显式指定的级别大于shebang声明，如果没有显式指定，那么就会按shebang声明的解释器执行

#

# &>，数据流重定向，将本该出现在屏幕上得输出内容，重定向到其他文件得内容中去，在这里将屏幕内容重定向到垃圾黑洞设备/dev/null，而不会在屏幕打印
# && 逻辑与，前一条命令执行成功，则与的命令紧接着开始执行，如失败则不执行
# || 逻辑或，前面的命令执行成功，则不执行后面语句，如失败，则执行后面的语句
# $? 的值表示最近一条命令是否成功，0:success，非零:fail