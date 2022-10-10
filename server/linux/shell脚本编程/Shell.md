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

### 0.2.4 bash shell 的环境配置文件

环境配置文件让bash在启动是直接读取这些配置文件，以规划好bash的操作环境。

配置文件又可以分为全体系统配置文件（`/etc/profile、/etc/bashrc`等）以及用户个人的偏好配置文件（`~/.bash_profile、~/.bashrc`），上面列举的文件是在登录shell的时候就会执行的配置。这两个配置文件是在退出登录的时候执行的`~/.bash_history、~/.bash_logout`。

login shell 与 non-login shell详见linux9.4.3

## 0.3 bash的常用知识点

1. 命令和文件自动补齐【Tab】： 

   - 在centos7当中，只要你安装了`bash-completion-2.7-5.el8.noarch`，你就有补全功能

   - ```bash
     [root@VM-4-8-centos ~]# rpm -qa | grep bash-comp
     bash-completion-2.7-5.el8.noarch
     ```

2. 命令历史记忆功能【上下键】：

   - 【!number】——执行history命令输出命令序列对应编号的命令
   - 【!string】——找到history历史执行命令中最近以string开头的命令、
   - 【!$】—— 上一个命令的最后一个参数
   - 【!!】——上一个命令
   - 【^R】——【ctrl + R】搜索历史命令

3. 命令别名

   - **alias  other_name='replace_operation'**
   - 取消别名：**unalias other_name**

4. 快捷键（ ^  —— ctrl ）：

   - ^R——搜索历史命令
   - ^D——logout，退出命令行
   - ^A ^E——光标移动到命令的开头，结尾
   - ^U ^K——从光标处删到开头、末尾
   - ^L—— clear 清屏
   - ^S  ^Q —— 锁定终端，使任何人不允许输入，但是输入操作会记录。解除ctrl +s的锁定，同时会展示或执行ctrl +s锁定时输入的指令

5. 前后台控制作业

   - & ——cmd &，命令丢到后台执行
   - nohup——nohup cmd，让你在脱机或注销系统后，还能够让工作继续进行
   - jobs -l ——查看作业，
   - ^c——是强制中断程序的执，程序不会再运行。^z——中断当前作业，维持挂起状态，
   - bg %jobnumber——后台运行jobs作业中jobnumber的作业，fg %jobnumber——前台运行jobs作业中jobnumber的作业
   - kill -signal %number
   - 通常在vim 编辑一个文档的时候，想干点其他什么事，这时候可以通过^Z去中断当前vim，做完之后，再通过fg回到之前中断的作业中来。

6. 重定向

   - 0——标准输入（<，<< EOF），1——正确输出（>覆盖内容，>> 追加内容），2——错误输出（2> 覆盖内容，2>>追加内容）
   - tee 双向重定向

7. 管道命令

8. 命令排序

   - 单行执行多个命令
   - `;`——不具备逻辑判断能力，每个命令都会执行，不论是否报错
   - `&& 逻辑与 ||逻辑或`——具备判断能力，
     - 逻辑与&&，当逻辑符号左侧的命令执行成功才会执行逻辑符号右侧的命令
     - 逻辑或||，当逻辑符号左侧的命令执行失败才会执行逻辑符号右侧的命令

9. 通配符

   ```bash
   # * 匹配任意多个字符
   ls ha*
   ls *tx*
   rm -rf *.pdf
   rm -rf *
   ll /dev/sd[a-z]*
   
   # ? 匹配任意一个字符
   touch love loove live l7ve; ll l?ve
   -rw-r--r-- 1 root root 0 Oct  3 15:24 l7ve
   -rw-r--r-- 1 root root 0 Oct  3 15:24 live
   -rw-r--r-- 1 root root 0 Oct  3 15:24 love
   # rm -rf l*ve
   
   # [] 字符集，匹配中括号中任意一个字符
   [a-zA-Z0-9] [^a-z]，尖角开头取非
   
   ll l[io]ve
   -rw-r--r-- 1 root root 0 Oct  3 15:28 live
   -rw-r--r-- 1 root root 0 Oct  3 15:28 love
   
   ll l[^a-Z]ve
   -rw-r--r-- 1 root root 0 Oct  3 15:28 l7ve
   
   # (cmd) 在子shell中执行cmd
   
   # {} 枚举值
   mkdir -pv ./{111/{aaa,bbb},222} || ls
   mkdir: created directory './111'
   mkdir: created directory './111/aaa'
   mkdir: created directory './111/bbb'
   111  222
   
   touch {a..c}{1..3} || ls
   a1  a2  a3  b1  b2  b3  c1  c2  c3
   
   cp -rv /etc/sysconfig/network-scripts/ifcfg-eth0 /etc/sysconfig/network-scripts/ifcfg-eth0.old
   # 可以写成下面的形式
   cp -rv /etc/sysconfig/network-scripts/{ifcfg-eth0,ifcfg-eth0.old}
   cp -rv /etc/sysconfig/network-scripts/ifcfg-eth0{,old}
   
   # \ 转义字符
   touch qin\ fei
   # 就创建了一个带有空格的文件
   # 而并没有创建两个文件
   
   # 转义回车符，使命令可以多行书写
   ls /etc/sysconfig/network \
   >/etc/hosts \
   >/etc/paaswd
   
   echo \\
   \
   echo "atb"
   # -e可以输出转义后的字符
   echo -e "a\tb"
   echo -e "a\nb"
   
   ```

10. echo 打印颜色文本。printf指令也可以格式化输出

    ```bash
    # 31m是指代文本的颜色，31m是红色，32m是绿色，33m是黄色。4开头的是修改文本的背景色。
    # 末尾一定要\e[0m，用于重置文本颜色，否则以后所有打印都是这个颜色
    echo -e "\e[1;31mthis is a red text.\e[0m"
    ```

11. 

# 1 变量

## 1.1 自定义变量

变量名：必须以字母或下划线开头，区分大小写。

变量赋值：a=1，注意：**等号两边不能有空格**

引用变量：`$var_name 或${var_name}`，`${var_name}`通常用在拼接字符串，变量的右侧有其他字符时。

## [字符串](https://blog.csdn.net/qq_59311764/article/details/121954162)

### 表示

字符串有三种表示：

1. 单引号

   ```bash
   # 1 单引号字符串中的变量是无效的，单引号里的任何字符（全部字符）都会原样输出
   a=1
   echo 'ab$acd'
   # 输出 ab$acd
   echo 'bb${a}bb'
   # 输出 bb${a}bb
   
   # 2 单引号字符串不能使用单独的一个单引号，对单引号使用转义符后也不行，
   echo 'abbbccc'ffff'
   line 9: unexpected EOF while looking for matching `''
   line 10: syntax error: unexpected end of file
   
   # 3 可以成对出现，作为字符串拼接使用
   echo 'abbbccc'aaa'ffff'
   # 输出 abbbcccaaaffff
   a=1
   echo 'ab'$a
   # 输出ab1
   echo 'ab'$a'ab'
   # 输出ab1ab
   ```

2. 双引号

   ```bash
   # 1 双引号可以有变量
   str='cd'
   str1="ab$str"
   echo $str1
   # 输出 abcd
   # 如果"ab$str"中$str后还有其他字符，就需要使用${}，如"ab${str}ef"，否则ef会被当做是变量名的一部分使用
   str2="ab${str}ef"
   # 输出 abcdef
   ```

3. 不用引号。

```bash
str1=abcd1
echo $str1
# 字符串拼接
echo 12$str1
echo 12${str1}34
```



### 单引号

```bash
# 1 单引号字符串中的变量是无效的，单引号里的任何字符（全部字符）都会原样输出
a=1
echo 'ab$acd'
# 输出 ab$acd
echo 'bb${a}bb'
# 输出 bb${a}bb

# 2 单引号字符串不能使用单独的一个单引号，对单引号使用转义符后也不行，
echo 'abbbccc'ffff'
line 9: unexpected EOF while looking for matching `''
line 10: syntax error: unexpected end of file

# 3 可以成对出现，作为字符串拼接使用
echo 'abbbccc'aaa'ffff'
# 输出 abbbcccaaaffff
a=1
echo 'ab'$a
# 输出ab1
echo 'ab'$a'ab'
# 输出ab1ab
```

### 双引号

```bash
# 1 拼接字符串
h="hello"
str1="1","${h}"
str2="1",$h
str3="1,${h}"
str4="1,$h"
echo $str1
echo $str2
echo $str3
echo $str4
# 全部输出一致
1,hello
```

