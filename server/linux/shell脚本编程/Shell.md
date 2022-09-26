# Shell

# 0 shell 初识

## 0.1 shell功能

![](.\legend\shell功能.png)

## 0.2 脚本初识

### 0.2.1 hello bash

```bash
#!/usr/bin/bash
ping -c1 www.baidu.com &> /dev/null && echo "baidu connected" || echo "baidu disconnected" 
echo $?

# #!/usr/bin/bash， #与!之间没有空格，这个叫shebang机制，程序的第一行，也仅第一行。声明这个脚本程序默认用哪个解释器执行，
# 如果通过 bash ./ping01.sh 执行，则就显式通过 bash 这个解释器执行，如果通过 ./ping01.sh 执行，则shebang机制就会生效，默认通过 #! 声明的解释器执行
# 显式指定的级别大于shebang声明，如果没有显式指定，那么就会按shebang声明的解释器执行

# &>，数据流重定向，将本该出现在屏幕上得输出内容，重定向到其他文件得内容中去，在这里将屏幕内容重定向到垃圾黑洞设备/dev/null，而不会在屏幕打印

# && 逻辑与，前一条命令执行成功，则与的命令紧接着开始执行，如失败则不执行
# || 逻辑或，前面的命令执行成功，则不执行后面语句，如失败，则执行后面的语句

# $? 的值表示最近一条命令是否成功，0:success，非零:fail
```

### 0.2.2 bash中临时执行其他语言脚本

通过重定向，标准输入，然后执行其他语言脚本

```bash
echo "hello bash"

/usr/bin/python <<- EOF
print "hello python"
EOF
# 命令执行重定向标准输入里的脚本

echo "hello bash"
```

### 0.2.3 脚本执行方式与父子shell

```bash
# script 的执行方式，细节见linux 9.7.1 script执行方式
# 1. 直接命令执行（文件必须有rx权限），绝对路径/相对路径/path路径下的脚本
# 2. bash进程执行
# 前两种，script执行完毕，bash内的数据都会被删除（子shell中生效）

# 3. 利用source执行，在父shell进程中执行，各项操作数据会在原来的bash中生效

# myshell.sh 内容如下
cd /home/
ls

chmod a+x myshell.sh
# 分别实验以下命令，在脚本执行完成后，查看当前shell环境（父shell）所在的目录位置
bash myshell.sh
./myshell.sh

. myshell.sh
source myshell.sh

# 结果
# 前两个父shell 的目录位置不变
# 后两个父shell 的位置跑到home下了
```

