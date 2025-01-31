# Shell

# 0 shell 初识

## 0.1 shell功能

![](.\legend\shell功能.png)

## 0.2 脚本初识

### 0.2.1 hello bash

注意：

1. 命令的执行从上而下，从左而右。
2. 命令、参数间的多个空白，空白行会被忽略，tab会视为空白
3. 读到enter就会执行，"\\[enter]"让一条命令扩展至下一行
4. #用来做批注

#### 1 第一个script

```bash
#!/bin/bash
# Program:
# This program shows ...
# History：
# 2005/08/23
PATH=/bin:/sbin:/usr/sbin/:/usr/local/sbin
export PATH
echo -e "Hello World! \a \n"
exit 0
```

1. **#!/bin/bash**：第一行是声明这个script使用的shell名称（**shebang机制**）
   - 通过这一句，当这个程序被执行时，它就能够加载bash的相关配置环境配置文件
2. **PATH**：
   - 主要环境变量的声明
   - 这里相当于一般程序里的**import**的功能
   - 建议务必要将一些重要的环境变量设置好（设置PATH与LANG等)，如此一来，则可让我们这个程序在进行时直接执行一些外部命令，而不必写绝对路径。
3. **echo**
   - 主要程序部分
4. **exit**
   - 告知执行结果。
   - 讨论一个命令执行成功与否，可以使用**$?**这个变量来看看。
   - exit n相当于一个return flag，我们可以通过查看$?这个flag来查看程序的执行情况如何
   - 系统自带的命令返回0代表执行成功，1代表失败



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

假设现在写了一个程序文件名是/home/scripts/myshell.sh

1. 直接命令执行：myshell.sh文件必须要具备可读与可执行**（rx）**的权限
   - 绝对路径：命令行直接输入：**/home/scripts/myshell.sh**
   - 相对路径：假设工作目录在/home/scripts，则命令行直接输入：**./myshell.sh**
   - 变量PATH的功能：将myshell.sh放到PATH指定的目录内，例如：/bin。然后在命令行直接输入：**myshell.sh**即可
2. 以bash进程来执行
   - 假设工作目录在/home/scripts，命令行直接输入：**bash myshell.sh** 或者 **sh myshell.sh**
3. 利用source来执行脚本
   - 假设工作目录在/home/scripts，命令行直接输入：**source myshell.sh**
4. 利用点命令来执行脚本
   - sourse 在`Bourne Shell`中的等价命令是一个点`.`，`source ./*.sh`和`. ./*.sh`的执行方式是等价的

**区别：**

1. 前二者：他们都会使用一个新的bash环境来执行script（**子进程的bash内执行**），script执行完毕后，bash内的数据都会被删除
2. 后者：**在父进程中执行**，各项操作数据都会在原本的bash内生效

![script执行方式的区别.jpg](../legend/script执行方式的区别.jpg)



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

### 0.2.5 script调试

**sh [ -nvx ] myshell.sh**

- n，不要执行script，仅检查语法的问题
- v，在执行script前，先将script的内容输出到屏幕
- x，将使用执行到的script内容显示到屏幕，+代表该行命令已执行

# 1 bash

![](../legend/硬核、内核与用户的相关性示意图.png)

只要能够操作应用程序的接口都能够称为shell，我们必须要通过“shell”将我们输入的命令与内核通信，好让内核可以控制硬件来正确无误的工作。

shell是看不见摸不着的，终端不是shell。

shell的版本众多，可以在**/etc/shells**文件中看到，我们可以使用哪些shell。

**“Bourne Again SHell（简称bash）”**就是比较流行的一个shell版本，bash是GNU计划中重要的工具软件之一。

## 1.0 bash初步

### 1.0.1bash功能

1. 命令记忆功能（history），前一次登录以前执行的命令记录在主文件夹内的**.bash_history**
2. 命令与文件补全功能（Tab），Tab按键用于补全
3. 命令别名设置功能（alias），eg：**alias lm=' ls -al '**
4. 作业控制、前台、后台控制（job control，foreground ，background）
5. 程序脚本（shell script），可以将平时管理系统常需要用的连续命令写成一个文件，该文件支持交互的方式
6. 支持统配符

### 1.0.2 bash内置命令判断：type

**type**：判断每个命令是否为bash的内置命令

- **type [-tpa] name**
- t，命令输出这些字眼：`file（外部命令）/alias（命令别名）/builtin（bash内置的命令）`
- p，如果name为外部命令式，才会输出完整的文件名
- a，会在path变量定义的路径中查找

### 1.0.3命令的执行

`\`反斜杠的应用在命令太长时，换行便于查看编辑

## 1.1 命令别名与历史命令

命令别名，自定义的变量在你注销bash后就会失效，所以你想要保留你的设置，就得将这些设置写入配置文件才行

### 1.1.1 命令别名

命令别名是一个很有趣的东西，特别是你惯用的命令特别长的时候

1. 设置别名**alias**：
   - **alias  other_name='replace_operation'**
   - alias定义规则与变量定义规则几乎相同
   - 替代既有命令：`alias rm='rm -i'`
   - 查看别名列表：`alias`
   - `eg：alias lm='ls -l | more'`，以后输入lm的命令就和`ls -l | more `的命令是一样的效果
2. 取消别名**unalias**：
   - **unalias other_name**
3. 

### 1.1.2 历史命令

**history**

- **n**：列出最近的n条命令
- **-c**：将目前shell中的所有history内容全部清除（当前登录的历史）
- -r：将histfiles中的内容读到目前的shell的history记忆中
- -w：将目前shell中history记忆内容更新到histfiles中，历史命令在我注销后，会将最近的HISTSIZE条记录到~/.bash_history中
- -a：将目前新增的history命令新增到histfiles中，若没有histfiles，则增加到**~/.bash_history**中

**历史命令执行**

- !!：执行上一个命令
- !n：执行第那条命令
- !command：由最近的命令向前搜寻以command开头的命令，并执行

## 1.2 bash shell操作环境

### 1.2.1 路径与命令查找顺序

命令执行顺序

1. 以相对/绝对路径执行命令，
2. 由alias找到该命令来执行
3. 由bash内置的（builtin）命令来执行
4. 通过$PATH这个变量的顺序找到第一个命令来执行

### 1.2.2 bash登录与欢迎信息

**登录之前显示**：/etc/**issue** 和 /etc/issue.net ：

- 前者/etc/issue：一个负责本地登录前显示， 是显示在TTY控制台登录前（非图形界面）
- 后者/etc/issue.net ：负责网络登录前显示。是显示在 Telnet (SSH默认不开启)远程登录前

登录之后显示（欢迎信息）：/etc/**motd**，不管你是 TTY 还是 PTS 登录，也不管是  Telnet 或 SSH 都显示这个文件里面的信息。

### 1.2.3 环境配置文件

环境配置文件让bash在启动是直接读取这些配置文件，以规划好bash的操作环境。

配置文件又可以分为全体系统配置文件以及用户个人的偏好配置文件

login shell与non-login shell

1. login shell ：取得bash时需要完整流程的，就称login shell，
   - eg1：你要由tty登录，需要输入账号和密码，此时取得的bash就称login shell
   - eg2： `su - user_name`
2. non-login shell：取得bash接口的方法不需要重复登录的举动，
   - eg1：在x window图形界面中启动终端机（此时没有再输入账号密码）
   - eg2：在原本的bash环境下在次执行bash这个命令（此时没有再输入账号密码），那第二个bash（子进程）也为non-login shell
   - eg3：`su user_name`

这两种情况会导致读取的配置文件不一样。

**non-login shell会读配置文件：**`~/.bashrc，而bashrc还会调用/etc/bashrc`

**login shell**会读取以下文件：

1. 系统整体配置

   - `/etc/profile`：login shell 必读，他还会调取外部设置数据，centos5.x会默认依序被调用进来
     - `/etc/inputrc`：自定义输入按键功能
     - `/etc/profile.d/*.sh`：这个目录下面的文件规定了bash操作接口的颜色、语系、ll与ls、vi、which的命令别名等**，如果想帮所有用户设置一些共享的命令别名可以在这个目录下自行创建扩展名为.sh的文件，并将所需写入即可**

2. 个人偏好设置

   - bash在读完整体环境配置后，接下来就会读取用户个人的配置文件
   - 依序读取`/~/.bash_profile > /~/.bash_login > /~/.profile`，前面的文件如存在，就不会再读后面的文件
   - `/~/.bash_profile会再调取/~/.bashrc`。**所以我们可以在此将自己的偏好设置写入该文件（`/~/.bashrc`）**

3. login_shell的配置文件读取流程

   ![](../legend/login_shell的配置文件读取流程.png)

**手动更新配置：source**

- **source 配置文件**

**其他相关配置文件**

- `/etc/man.config`：执行man时man page的路径到哪里去找
- `~/.bash_history`：历史命令
- `~/.bash_logout`：当我注销bash后系统再帮我做完什么操作才离开

### 1.2.4 终端机快捷键

**快捷键设置**

查看终端机**"特殊功能"**按键列表：**stty -a**，stty（setting tty）

输出的**^**代表【Ctrl】的意思，其他的eof，erase，intr等自行查阅便知

**按键修改**，eg：**stty erase ^h**

修改终端机设置值，**set [ -uvCHhmBx ]**

**常用快捷键**

| 组合键        | 意义                                                     |
| ------------- | -------------------------------------------------------- |
| Ctrl + C      | 终止当前输入的命令，终止运行的程序                       |
| Ctrl + D      | 退出bash子进程，or 输入结束（eof），从光标删右方一个字符 |
| Ctrl + Z      | j将运行的程序送到后台                                    |
| Ctrl + L      | q清屏                                                    |
| Ctrl + R      | s搜索历史命令，搜索到回车即可执行                        |
| Ctrl + U or K | c从光标位删至行首or行尾                                  |
| Ctrl + S or Q | 暂停or 恢复屏幕输出                                      |



### 1.2.5 通配符与特殊符号

| 通配符       | 意义                           |
| ------------ | ------------------------------ |
| *****        | 0到多个任意字符                |
| **？**       | 至少一个任意字符               |
| **[ ]**      | 字符集，eg：[abceft]           |
| **[ - ]**    | 字符段，eg：[a-z]              |
| **[ ^ ]**    | 取非，eg：`[^a-z]`，非小写字母 |
|              |                                |
| **特殊符号** | **意义**                       |
| #            | 批注                           |
| \            | 转义                           |
| **\|**       | 管道符                         |
| **;**        | 命令分割符                     |
| **~**        | 用户主文件                     |
| **$**        | 变量前导符                     |
| **&**        | 作业控制                       |
| **!**        | 非                             |
| **/**        | 目录符号，路径分割符           |
| **>,>>**     | 数据流重定向，输出导向         |
| **<,<<**     | 数据流重定向，输入导向         |
| **''**       | 不具占位符                     |
| **""**       | 具占位符                       |
| **``**       |                                |
| **( )**      | 子shell的起始与结束            |
| **{ }**      | 命令块组合                     |
|              |                                |
|              |                                |

## 1.3 数据流重定向

数据流重定向就是将某个命令执行后应该要出现在屏幕上的数据传输到其他地方去，例如文件或设备

![](../legend/命令执行过程的数据传输情况.png)

1. **标准输入stdin**：需要由键盘输入的数据改由文件内容来替代，代码为**0**，使用 <（指定内容输入文件） 或 <<（设置终止符）；
2. **标准输出stdout**：命令执行所回传的正确信息，代码为**1**，使用 **>（覆盖内容） 或 >>（追加内容）**
3. **标准错误输出stderr**：命令执行失败后，所传回的错误信息，代码为**2**，使用 2>（覆盖内容） 或 2>>（追加内容）



```bash
####标准输出####

#屏幕不输出任何信息，~下有一个mine文件创建。
ls -al / > ~/mine 
#若~下mine文件不存在，则创建
#若存在，则清空文件，在写新数据
# >> 追加数据到原文件

#将stdout写入list_right文件，将stderr写入list_error
find /home -name .bashrc > list_right 2> list_error

#stdout与stderr写入同一个文件
find /home -name .bashrc > list 2>&1
# 混合重定向 &>
find /home -name .bashrc &> list

#垃圾黑洞设备/dev/null
find /home -name .bashrc 2> /dev/null

####标准输入####

cat > introduce << "eof"
my name is lxx, hen gao xin ren shi da jia
eof
#eof作为文本输入的结束，而不需要按Ctrl+d

#将introduce的内容复制到catfile文件中
cat > catfile < introduce
```

### 命令执行的判断

很多命令我想要一次输入去执行，而不想要分次执行。除了通过shell script，可以通过 **;  &&  ||**

- 如若命令之间**无相关性**可以通过命令间间隔**`;`**分号，
  - **cmd1 ; cmd2 ; cmd3**
- 如若命令之间**有相关性**可以通过 **`&&  ||`**来实现
  - **cmd1 && cmd2**，cmd1执行成功才执行cmd2
  - **cmd1 || cmd2**，cmd1执行失败才执行cmd2
  - cmd1 || cmd2 && cmd3，cmd1执行成功后执行cmd3，cmd1执行失败执行cmd2，cmd2执行成功后执行cmd3

## 1.4 管道命令（pipe）

Shell 还有一种功能，就是可以将两个或者多个命令（程序或者进程）连接到一起，把一个命令的输出作为下一个命令的输入，以这种方式连接的两个或者多个命令就形成了**管道（pipe）**。

Linux 管道使用竖线**`|`**连接多个命令，这被称为管道符。

语法格式：**`command1 | command2 [ | commandN... ]`**

![](../legend/管道命令的处理示意图.png)

命令输出的数据需要经过几个连续命令的处理，才是我们想要的数据，就会用到管道命令。

**注意**：

- 管道命令仅会处理stdout，对stderr会予以忽略
- 管道命令必须要能够处理来自前一个命令的stdout作为自己的stdin才行。
- 在管道命令里**“  -  ”**可以表示stdin或者stdout

### 1.4.1 选取命令

一般来说，选取信息通常是针对**“行”**来分析的，并不是整篇信息分析的。

1. **cut**
   - 从一行信息中取出我们想要的某些段
   - **cut  -d  'divider'  -f  field1 [, field2, field3 ] **：d 后面接分隔符（' ' 表示空白符），f 后面接我们以分隔符分割的第几段，用整数n
   - **cut -c n1-[n2**]：以字符数为单位，显示字符区间的内容
2. **grep**
   - 分析一行的信息，若有我们想要的信息，则取出该行
   - **grep [ -acinv ] [ --color=auto ]  '查找字符串'  filename**
   - i，忽略大小写
   - n，顺便输出行号
   - v，反向选择，即显示出没有"查找字符串"的行
   - c，计算“关键字”在所在行出现的次数
   - filename，可由通配符代替，执行多文件查找信息工作

### 1.4.2 排序去重统计命令

计算数据里相同类型的数据总数，类似于数据库里的聚合函数般。

1. **sort**
   - **sort[ -tkuf ] filename**
   - t，分隔符，默认tab
   - k，以分割符分割的第几个字段排序
   - u，相同数据取其一，去重
   - f，忽略大小写
2. **uniq**
   - 内容排序完成，去重
   - uniq [ -ic ]
     - i，忽略大小写
     - c，重复项计数
3. **wc**
   - 统计内容数据（word count）
   - **wc [-lwm]**
   - l，列出行数
   - w，流出单词数
   - m，列出字符数

### 1.4.3 字符转换命令

比如说将内容中大写改小写，tab转空格键

1. **tr**
   - 删除或替换
   - **tr [ -ds ] SET_OR_STR**
   - d，删除
   - s，替换
2. **col**
   - **col [ -xb ]**
   - x，将tab转换成对等的空格键
3. **join**
   - 处理两个文件之间的关联数据，将两个文件中有相同数据的那一行续接在一起。类似于数据库中的**关联查询**
   - **join [ -ti12 ] file1 file2**
   - t，分隔符，默认以空格，并且对比两个文件的第一个字段
   - i，忽略大小写
   - 1，第一个文件用哪个字段
   - 2，第二个文件用哪个字段
4. **paste**
   - 直接将两个文件的同行，连接在一起
   - **paste [-d] file1 file2**
   - d，分割符，默认tab
   - -，如果file写成-，表示stdin

### 1.4.4 切割命令

如果一个文件太大，导致一个携带式设备无法复制的问题，通过split，就可以将一个大文件依据文件的**大小或行数分割成小文件**

**split [-bl] file prefix_name**

- b，按文件内存大小进行分割，例如b，k，m等
- l，按行数进行分割
- prefix_name，小文件名的前缀

### 1.4.5 参数代换

xargs可以产生某个命令的参数，xargs 可以读入stdin的数据，并且以空格符或断行字符进行分辨，将stdin的数据分割成arguments。

**xargs [ -0epn ] command**

- 将stdin的内容经xargs分析后，处理成参数（或者多组参数），一组参数作为command的参数使用
- eof，停止分析符
- n，多少个参数为一组
- p，每输入一组参数执行command前询问用户是不是要执行，执行输入y

### 1.4.6 双向重定向

**tee**

- **tee会同时将数据流送予文件与屏幕**
- **tee [-a] file**
- a，以累加（append）的方式

## 1.5 其他

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

   - & ——cmd &，命令丢到后台执行，
   - nohup——nohup cmd，让你在脱机或注销系统后，还能够让工作继续进行
   - jobs -l ——查看作业，
   - ^c——是强制中断程序的执，程序不会再运行。^z——中断当前作业，维持挂起状态，
   - bg %jobnumber——后台运行jobs作业中jobnumber的作业，fg %jobnumber——前台运行jobs作业中jobnumber的作业
   - kill -signal %number
   - 通常在vim 编辑一个文档的时候，想干点其他什么事，这时候可以通过^Z去中断当前vim，做完之后，再通过fg回到之前中断的作业中来。

6. [重定向](https://www.milinger.com/a250.html)

   - 0——标准输入（<，<< EOF），1——正确输出（>覆盖内容，>> 追加内容），2——错误输出（2> 覆盖内容，2>>追加内容）
   - &>——将正确输出和错误输出**都**重定向到文件
   - tee 双向重定向

7. 管道命令

   - command1 | command2

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

10. [echo 打印颜色文本。printf指令也可以格式化输出](https://blog.csdn.net/XYliurui/article/details/102761476?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168078941816800182792936%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168078941816800182792936&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-102761476-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=echo%20%E8%BE%93%E5%87%BA%E7%BB%BF%E8%89%B2&spm=1018.2226.3001.4187)

    ```bash
    # 31m是指代文本的颜色，31m是红色，32m是绿色，33m是黄色。4开头的是修改文本的背景色。
    # 末尾一定要\e[0m，用于重置文本颜色，否则以后所有打印都是这个颜色
    echo -e "\e[1;31mthis is a red text.\e[0m"
    ```

11. 

# 2 shell变量

变量的类别：自定义变量和环境变量

## 2.1 变量

### 2.1.1 显示与设置

**set——查看所有变量（包含环境变量与自定义变量）**

1. 变量的引用：**$var_name**

   - **echo $变量名**
   - **echo ${变量名}**，常用于数字参数的变量，以及对变量进行删除替换等操作时（相当于计算变量）
   - echo 功能有很多，这里只是用到了查看变量的功能
   - 列出当前环境变量——env
   - 列出所有变量（包含环境变量与自定义变量）——set

2. 变量的赋值

   - 显式赋值，格式：**var_name=var_val**

     ```bash
     #1.常量赋值
     ip1=192.168.4.25
     address="beijing tiananmen"
     #2.变量间赋值
     var1=${ip1}
     var2="ip is ${var1}"
     var3="ip is $var2"
     #3.命令替换，先执行命令，再赋值，反引号``等价于$()
     # 反引号的命令替换
     today=`date +%F`
     echo $today # 打印2022-01-14
     touch `date +%F`_file.txt
     
     # $()的命令替换
     today1=$(date +%F)
     echo $today1 # 打印2022-01-14
     
     ```

   - 键盘读入：**read** 

     - `read [ -ptn ] var_name`
     - p，后面接输入提示文本
     - t，输入等待时间，记住输入后一定要按回车，否则时间结束后依旧不能声明变量
     - n，只获取输入的n个字符
     
     ```bash
     read -p "请输入你需要备份的文件名: " back_file
     read ip1 ip2 ip3
     #输入 1.1.1.1 2.2.2.2 3.3.3.3，空格相隔，可以一次赋多个值
     
     #!/usr/bin/bash
     read -p "请输入姓名，性别，年龄【eg：qqq m 25】：" name sex age
     echo "你输入的姓名：$name，性别：$sex，年龄：$age"
     ```
     
   - 

   - 规则：

     - **等号两边不能直接接空格**，变量命名只能是英文和数字（数字不能开头）

     - 引用变量：`$var_name 或${var_name}`，`${var_name}`也用在拼接字符串（解决歧义），变量的右侧有其他字符时。

     - 变量的值若有空格，可用双引号或单引号将值罩起来。

       - 双引号可以解析值中的变量（占位符），而单引号不行

         ```bash
         qi=kh
         mine="my name is $qi" 
         echo $mine
         # my name is kh
         mine='my name is $qi' #重新赋值
         echo $mine
         # my name is $qi
         ```

     - 特殊符号用反斜杠`\`转义

     - **追加变量内容**

       ```bash
       PATH="$PATH":/home/bin
       ```

     - **环境变量**（全局变量）：**export var_name**

3. **取消变量unset**

   - **unset var_name**

### 2.1.2 变量运算

#### 1 整数运算

```bash
# 1 expr，
# 加减乘除取余，+ - \* / % ，其中乘号需要转义，因为会误认为*为通配符
expr 1 + 3
# 输出 4
num1=10
num2=20
expr $num1 + $num2
# 输出 30
sum=`expr $num1 + $num2`
echo $sum #30
# 乘
expr $num1 \* $num2
# 输出 200

# 2 $(())
echo $(($num1 * $num2))
echo $((num1 * num2)) # 括号里面可以不用变量引用符
echo $((5-3*2))
echo $(((3+2)*5))
echo $((2**10))
sum=$((num1 + num2))

# 3 $[]
echo $[2+2]
echo $[2**10]
num1=$[2**10]
echo $num1

# 4 let
# 运算间不能加空格，必须写一起
let num3=1+2
echo $num3
no=0
let no++ # 自加1操作
let no-- # 自减1操作
let no+=10 # no=$no+10
let no-=20 # no=$no-20

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
```

#### 2 小数运算

bash 不支持浮点运算，如果需要进行浮点运算，需要借助bc,awk 处理。

有内建的bash计算器，称作bc。在shell提示符下通过bc命令访问bash计算器。

```bash
echo "scale=2;6/4" | bc
awk 'BEGIN{print 1/2}'

# bc命令能识别输入重定向，
# EOF文件字符串标识了内联重定向数据的开始和结尾。记住仍然需要反引号来将bc命令的输出赋给变量。
#!/bin/bash
var1=10.46
var2=43.67
var3=33.2
var4=71
var5=`bc << EOF
scale = 4
a1 = ($var1 * $var2)
b1 = ($var3 * $var4)
a1 + b1
EOF`
echo The final answer for this mass is $var5
```



## 2.2 环境变量env

### 2.2.1 变量列表

**env——查看所有环境变量**

**常见环境变量**

- HOME：代表用户主文件夹的位置，通过cd ~或者直接cd就可以切换到这个文件夹
- SHELL：当前这个环境使用的shell是那个程序
- PATH：就是执行文件查找的路径，目录与目录间通过冒号**":"**相隔
- [IFS](https://blog.csdn.net/whatday/article/details/122508281)：internal field separator，字段分割符
  - IFS=$'\n'  表示用 换行符 做分隔
  - IFS="\n" 与 IFS='\n'  都是用 n 字符作为分隔
- $BASH_SUBSHELL：子shell的个数
- 

```bash
# 关于IFS，经常在读文件，做for循环时会用到。IFS它用在自定义分割字段
# 在shell中，修改IFS后，记得还原

#!/bin/sh
 
conf="ABC
A B C
1|2|3
1 2 3"
echo "$conf"
 
echo --------------
echo IFS:
echo -n "$IFS"|xxd # xxd能将一个给定文件或标准输入转换为十六进制形式
echo --------------
for c in $conf;do
    echo "line='$c'";
done
 
echo --------------
#IFS=$'\n'  表示用 换行符 做分隔
#IFS="\n" 与 IFS='\n'  都是用 n 字符作为分隔
IFS=$'\n'
echo IFS:
echo -n "$IFS"|xxd 
echo --------------
for c in $conf;do
    echo "line='$c'";
done

# -----------------输出结果-----------------
# ABC
# A B C
# 1|2|3
# 1 2 3
# --------------
# IFS:
# 00000000: 2d6e 2020 090a 0a                        -n  ...
# --------------
# line='ABC'
# line='A'
# line='B'
# line='C'
# line='1|2|3'
# line='1'
# line='2'
# line='3'
# --------------
# IFS:
# 00000000: 2d6e 200a 0a                             -n ..
# --------------
# line='ABC'
# line='A B C'
# line='1|2|3'
# line='1 2 3'
```



**set——查看所有变量（包含环境变量与自定义变量）**

#### 预定义变量

| 符号   | 意义                                                         |
| ------ | ------------------------------------------------------------ |
| **$0** | 命令的名字（位置参数的一种），带路径的命令名字               |
| **$*** | 所有的参数，<br />参数的数组类型，当有多个参数的时候，每个参数占用一个数组元素。 |
| **$@** | 所有的参数，<br />参数的字符串类型，当有多个参数的时候，所有参数拼成一个长字符串作为一个参数。<br />[$*与$@的区别](https://www.jianshu.com/p/6bed59a8eeda) |
| **$#** | 参数的个数，                                                 |
| **$$** | 当前进程的PID                                                |
| **$!** | 上一个后台进程的PID                                          |
| **$?** | 上一个命令的返回值，0代表上一个命令执行成功                  |

```bash
echo "所有参数\$*：$*"
echo "所有参数\$@：$@"
echo "参数个数\$#：$#"
echo "当前进程PID\$\$：$$"

$ ./预定义变量.sh a b c d e

所有参数$*：a b c d e
所有参数$@：a b c d e
参数个数$#：5
当前进程PID$$：1757

```



- **PS1**：命令提示符，在命令之前的部分，也可以修改，可以查看相关资料修改

  ```bash
  [root@iZbp1a1nnstgr4w2bu73ovZ ~]# cd ../etc
  ```

- **$：关于本shell的线程号PID**

- **?：关于上个命令的回传码**

- OSTYPE，HOSTTYPE，MACHTYPE：主机硬件与内核等级

### 2.2.2 自定义变量转环境变量

**export var_name**

环境变量与自定义变量的区别在于：该变量是否会被子进程所继续引用。

当你登录linux并取得一个bash后，你的bash就是一个独立的进程。

在一个bash（父进程）中执行另一个bash（子进程），父进程就会处于暂停sleep的情况，只有子进程结束或被exit才能回到父进程中。

**子进程会继承父进程的环境变量，而不继承自定义变量**

**变量作用域**：被export后的变量，我们称为环境变量（全局变量）。环境变量可以被子进程（子shell）所引用，但是自定义变量（局部变量）则不会存在于子进程中。

**多个shell脚本中使用公共的自定义变量的方案：**

1. 用一个sh01.sh专门存放公共的变量
2. 其他的shell脚本在使用公共变量之前，执行**source  path/sh01.sh**（不会启用子shell，在当前shell声明自定义变量）
3. 即可引用

### 2.2.3 添加环境变量

```bash
# 1.针对所有用户长久有效
# 通过export添加变量参数
vim /etc/profile

# 通过这个函数可以实现在现有的path上面添加新的path
pathmunge () {
    case ":${PATH}:" in
        *:"$1":*)
            ;;
        *)
            if [ "$2" = "after" ] ; then
                PATH=$PATH:$1
            else
                PATH=$1:$PATH
            fi
    esac
}
pathmunge /usr/local/nginx/sbin
export PATH

# 保存etc/profile后，需要下面的命令使文件生效
source /etc/profile

# 2.针对当前用户长久有效
vi ~/.bash_profile
# 操作同上

# 3.针对当前登录会话有效，关闭之后下次登录shell就无效了
# 直接将上面的export 命令执行一遍。这种方式最简便，适合临时设置环境变量使用
```

## 2.4 命令的位置参数

这个概念就如同javascript，或python函数的参数列表一样，只不过在bash中这个函数概念换成了命令。当然bash中也有函数这个概念，后面会说到。

shell script 可以在脚本文件名后面带参数。而不需要像read命令在执行过程中再去输入参数。

eg：**/~/scripts/myshell.sh param1 param2 param3 param4**

在script脚本中可以使用以下变量去引用位置参数的值

- $0：这个变量用于获取执行脚本的名字
- $#：获取参数的个数
- $@：获取整个参数字符串，每个参数间通过空格键分割，等价于`$*`
- $n：n为位置的索引，$1，$2

**shift未知参数的移除和偏移**

**shift [n]**，这个就如同javascript数组的shift函数功能，用于移除位置参数序列的前n个元素。

```bash
while [ $# -ne 0 ]
do
	useradd $1 # 添加用户，./script.sh alice qqq kkk hhh
	# let sum+=$1 # 所有参数求和，./script.sh 1 2 3 4
	shift 1 # 执行后，命令后面的第一参数会被踢出参数队列，而执行前的第二个参数被推向了参数队列的首位
done
```



## 2.5 declare声明变量类型

- **declare**

  - `declare [-aAixr] var_name`

  - a，array，声明为数组变量

  - A，关联数组

  - i，integer，声明为整形变量
  
    ```bash
    declare -i total=100+300+50
    echo $total
    #450
    ```

  - x，声明为环境变量，+-x，可以变换作用域
  
  - r，设置为只读

## 2.6 array变量

```bash
var[1]="first 1"
var[2]="second 2"
var[3]='last 3'
echo ${var[1]},${var[2]},${var[3]}

```

## 2.7 变量内容的删除、替代与替换、追加

**删除与替代，对原变量没有影响**

| 变量设置方式                         | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| ${#var}                              | 获取var变量所代表的字符串长度                                |
| ${var#关键字}                        | 变量内容从开头匹配"关键字"，将符合的最短数据删除             |
| ${var##关键字}                       | 从头，最长匹配删除                                           |
| ${var%关键字}                        | 从尾，最短匹配删除                                           |
| ${var%%关键字}                       | 从尾，最长匹配删除                                           |
| ${var:start[:legth]}                 | 从start的索引开始，切一个length长度的字符串                  |
| ${var/旧子串/新子串}                 | 变量内容符合旧子串，替换第一个被匹配到的旧子串为新子串       |
| ${var//旧子串/新子串}                | 匹配旧子串，将变量内容中的所有旧子串替换为新子串             |
| **var=${var1 [ : ] [ +-= ]  expr }** | 测试后，赋值<br />在某些时刻我经常需要判断某个变量是否存在，若存在，使用既有设置，若不存在则给予一个新值。<br />${var1-expr}，var1若不存在，则返回expr的值<br />**冒号:**能识别str是否为空字符串`''`，将空字符串也认定为变量str未设置。<br />**减号-**为str为未设置（未定义）时，起到赋值作用<br />**加号+**为str为已设置时，起到赋值作用<br />**等号=**与减号的作用相似，**只是赋值的作用会影响到str** |
| PATH="$PATH":/home/bin               | 追加内容                                                     |

```bash
#!/usr/bin/bash
url=www.sina.com.cn
# 变量的长度
echo ${#url}
# 1.删除
echo ${url#*.}
# 输出 sina.com.cn
echo ${url##*.}
# 输出 cn
echo ${url%.*}
# 输出 www.sina.com
echo ${url%%.*}
# 输出 www

# 2.索引和切片
# 从字符串的第0个开始，切长度为5个串
echo ${url:0:5}
# 输出 www.s
# 从字符串的第5个开始，切长度为5个串
echo ${url:5:5}
# 输出 ina.c
# 从字符串的第5个开始，切到末尾
echo ${url:5}
# 输出 ina.com.cn

# 3.替换
# 替换第一个被匹配的字符
echo ${url/n/N}
#www.siNa.com.cn
# 替换所有被匹配到的字符
echo ${url//n/N}
# www.siNa.com.cN
```



**变量的测试与内容的替换**

在某些时刻我经常需要判断某个变量是否存在，若存在，使用既有设置，若不存在则给予一个新值。

**冒号:**能识别str是否为空字符串`''`，将空字符串也认定为变量str未设置。

**减号-**为str为未设置（未定义）时，起到赋值作用

**加号+**为str为已设置时，起到赋值作用

**等号=**与减号的作用相似，**只是赋值的作用会影响到str**

**var=${str [ : ] [ +-= ]  expr }**

**追加内容**

```bash
PATH="$PATH":/home/bin
```

## 2.8 字符串

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
   
   # 2 拼接字符串
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

3. 不用引号。

```bash
str1=abcd1
echo $str1
# 字符串拼接
echo 12$str1
echo 12${str1}34
```

其他

```bash
# 1 获取字符串的长度
name1="hello 1"
echo ${#name1}
# 输出 7

# 2 提取子字符串
name="hello123"
echo ${name:2:5}
```

## 2.9 一般符号

| 符号      | 含义                      |
| --------- | ------------------------- |
| **()**    | 在子shell中执行，定义数组 |
| **(())**  | 数值比较                  |
| **$()**   | 命令替换                  |
| **$(())** | 整数运算                  |
| **{}**    | 集合，eg：{1..254}        |
| **${}**   | 变量引用                  |
| **[]**    | 条件测试                  |
| **[[]]**  | 条件测试，支持正则        |
| **$[]**   | 整数运算                  |

**执行脚本**

| 执行方式            | 解释                              |
| ------------------- | --------------------------------- |
| path/文件.sh        | 需要执行权限，在子shell中执行     |
| bash path/文件.sh   | 不需要执行权限，在子shell中执行   |
| . path/文件.sh      | 不需要执行权限，在当前shell中执行 |
| source path/文件.sh | 不需要执行权限，在当前shell中执行 |

**调试脚本**

| 命令             | 解释                                                         |
| ---------------- | ------------------------------------------------------------ |
| **sh -n 02.sh**  | 仅调试syntax error                                           |
| **sh -vx 02.sh** | 以调试的方式执行，查询整个执行过程<br />执行该命令后，窗口中打印的结果中，+代表已经执行的动作 |

# 3 程序结构

## 3.1 判断式

条件测试语句

判断式就如同其他程序语言中用于输出true or false的条件判断句，然而在bash中是通过执行命令，命令返回**$?**这个变量来判断的。

这里还有其他一些常用的判断式

### 3.1.1 判断命令test

**变量或常量字符串在判断式中最好用双引号括起来，括起来后，变量如果为空或未定义，不会报语法错误，并且长度都按0处理**

**test expression**

1. 判断文件名的类型，
   - **test [ -efdbcSpL ] filename**，
   - e—filename是否存在，f一判断是否存在并为file类型，d—directory，b—block device，c一character device，S一socket文件，L—连接文件，p—pipe文件
2. 检验文件的权限
   - **test [ -rwxugks ] filename**
   - r一判断文件是否存在并具有可读权限，w一可写，x—可执行，u一SUID属性，g—SGID属性，s—非空白文件
3. 两个文件之间比较
   - **test file1 [-nt ot ef] file2**
   - nt—newer than判断file1是否比file2更新，ot一older than，ef一判断是否为同一个文件
4. 关于两个整数之间的判定
   - **test n1 [-eq nq gt lt ge le] n2**
   - eq一equal，ne一not equal，gt一greater than，lt一less than，ge—greater or equal，le—less or equal
5. 判定字符串的数据
   - **test [-zn] string**
   - z一若为为空字符串，则true，n一若为非空字符串，则true
   - **test str1[!]=str2**
   - 判断两个字符串是，否（!）相等
6. 多重条件判定
   - **test cond1 [-ao] cond2**
   - a—and 两个条件必须同时成立，则为true。o—or 任一条件成立，则为true
   - **test ! cond**
   - 条件取非
   - 
7. 

### 3.1.2 判断符号命令——左中括号[

**test expression** 可以被 **[ expression ]** 替换

左中括号（ **`[`** 是一个命令）就相当于test，上面的命令判断语句，可以通过让中括号括起来而去掉test关键字，形成判断句。

中括号常用在条件判断式中if...then...fi中

注意：

1. **中括号[ ]内的每一个组件都需要有空格键来分割**，

   - ```bash
     # 我们可以发现左中括号[是一个命令
     which [
     /usr/bin/[
     # 而右中括号]不是一个命令
     which ]
     /usr/bin/which: no ] in (/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/root/bin)
     ```

   - 所以左中括号是一个命令，相当于test，左中括号后面的，直到右中括号之前，包括右中括号，都是左中括号的参数

   - 所以左中括号之后，参数与参数之间，都必须空格相隔

   - 注意：中括号与expression之间必须有空格，否则会报**command not found**

2. **中括号内的变量和常量，都需要用双引号括起来（否则有可能会出现一些意想不到的错误）**

```bash
eg:
[ "$HOME" == "/bin" ]
# 当unset home之后，再执行上面的语句将会报语法报错，如果用双引号括起来，则不会出现语法错误。
```

## 3.2 选择结构

### 3.2.1 if语句

条件书写：

```bash
[cond1 -a cond2 ] && [ cond3 ] || [ cond4 ]
```

```bash
#单判断条件语句
if [ cond1 ]; then
	程序段1
else
	程序段2
fi #条件语句结束

if [cond1]
then		# 因为 if [cond1] 和 then 都是命令，所以多个命令放在一行需要加分号";"，如果换行则没必要加分号。
	程序段1
else

#多判断条件语句
if [ cond1 ]; then
	程序段1
elif [ cond2 ]; then
	程序段2
else
	程序段3
fi #条件语句结束
```

条件测试不一定需要**test或[]**，**if后面可以跟任何一个可以返回真假的语句，它会根据语句的返回值，来确定是成功还是失败**

```bash
read -p "请输入用户名： " user
if id $user &>/dev/null; then 
	# 上面一行等价于下面两行
    # id $user &>/dev/null
    # if [ $? -eq 0 ]; then
    echo "用户已存在"
else
    useradd $user
    if [ $? -eq 0 ]; then
        echo "$user 已创建成功"
    fi
fi

```

### 3.2.2 case语句

case语句只能做字符串比较，不能做大小比较

```bash
case $var in
"state1")
 	程序段1
 	;;
"state2")
 	程序段2
 	;;
"state3" | "state4")
 	程序段2
 	;;
*)#用*表示其他状态
 	程序段n
 	;;
esac
```

```bash
#!/usr/bin/bash
read -p 'please input username: ' $user
if [ ! -z "$user" ]; then
    echo "input error"
    exit
fi

id $user &>/dev/null
if [ "$?" -ne 0 ];then
    echo "$user is unexited"
    exit
fi

read -p "are you sure? [y/n]" action

# if语句的写法
# if [ "$action" = "y" -o "$action" = "Y" -o "$action" = "yes" -o  "$action" = "YES" ]; then
#     userdel -r $user
#     echo "$user is deleted successfully"
# else
#     echo "action is canceled"
# fi

# case语句的写法
case $action in
"y" | "Y" | "yes" | "YES")
    userdel -r $user
    echo "$user is deleted successfully"
    ;;
*)
    echo "action is canceled"
    ;;
esac
```

### 3.3.3 练习

1. 拼指定主机测试，
2. 判断用户是否存在
3. 判断当前内核版本是否为3，且次版本是否大于10
   - uname -r
4. 判断vsftpd软件包是否安装，如果没有安装则自动安装
   - rpm -q vsftpd
5. 判断httpd是否运行
6. 报警脚本
   - 根分区剩余空间小于20%
   - 内存已使用空间大于80%
   - 想用户alice发送邮件
   - 配合例行性任务crond每5分钟执行一次
7. 

## 3.3 循环结构

### 3.3.1 for循环

```bash
#foreach循环
# for 循环默认按照空格来分割序列，除非修改了环境变量IFS，
for var in constant1 constant2 constant3
do
	程序段$var可以获取constant列表的内容
done

#for循环
s=0
for ((i=0; i<$var; i=i+1 ))
do
	s=$(($s+$i))
done
```

创建序列的方式：

- **{start..end}**，这种方式start和end不支持变量，只能是常量
- **seq start end**，推荐，-w还可以做等位补齐功能
- 读文件

循环体丢到后台（**并发**）执行：**{循环体}&**

等待循环所有执行内容在后台全部完全结束：**wait**

```bash
# 1.序列循环
#!/usr/bin/bash
# 这里有两种方式创建一个序列，一个是{start..end}，一个是seq start end
for i in {2..254} # 不支持动态变量的改变序列长度
# for i in `seq 2 254` # seq -w 100 w参数可以等位补齐，例如：001,002...100
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
```

```bash
# 2.读文件，然后循环
# 不推荐用for来处理文件，因为for默认按照空格来分割序列，需要修改IFS（internal field separator）
for ip in `cat ip.txt`
do
    # echo $ip
    ping -c1 $ip &> /dev/null
    if [ $? -eq 0 ];then
        echo "${ip} is up."
    else
        echo "${ip} is down."
    fi
done
```

```bash
# 3. 通过文件，批量创建用户，用户名和密码都在文件中
# 需要改分割符IFS

# 判断参数个数
if [ $# -eq 0 ];then
    echo "must input file parameters"
fi

# 判断文件是否存在
if [ ! -f $1 ]; then
    echo "file：$1 is unexisted"
fi

# 我们希望文件内容按回车分割，而不是按空格或tab分割
# 这里需要重新设置IFS，使用完毕后需要还原
IFSTemp=$IFS
IFS=$'\n'
# user.txt 内容
# abc 145566
# ttt 123456
# dfd 145987
for line in `cat user.txt`
do
    # 如果是空行，跳过
    if [ ${#line} -eq 0 ]; then
        continue
    fi
    # 分割每行内容
    user=`echo "$line" | awk '{print $1}'`
    pass=`echo "$line" | awk '{print $2}'`
    id $user &>/dev/null
    if [ $? -eq 0 ]; then
        echo "$user already exists"
    else
        useradd $user
        echo "$pass" | passwd --stdin $user &>/dev/null
        if [ $? -eq 0 ]; then
            echo "$user created successfully"
        fi
    fi
done
IFS=$IFSTemp
```

```bash
# 如果这里不要in以及后面的序列，那么它默认会获取整个脚本（或函数）的参数列表
for i
do
	let sum+=i
done
```



### 3.3.2 while循环

条件为真，执行循环。

for循环需要读文件时需要修改分割符（默认空格和tab），而while没有这个问题，while默认换行符。

逐行处理文件，请优先考虑while循环，而不是for。

```bash
#while循环
while [ condition ]
do
	程序段
done
```

```bash
i=1
sum=0
while [ $i -le 100]
do
	let sum+=$i
	# 等价于
	# let sum=$sum + $i
	let i++
done
printf "sum=${sum}"
```



```bash
# 通过文件批量创建用户
while read line
do
    # 不需要做空行判断，因为read 空行会得到1的返回值
    # 分割每行内容
    user=`echo "$line" | awk '{print $1}'`
    pass=`echo "$line" | awk '{print $2}'`
    id $user &>/dev/null
    if [ $? -eq 0 ]; then
        echo "$user already exists"
    else
        useradd $user
        echo "$pass" | passwd --stdin $user &>/dev/null
        if [ $? -eq 0 ]; then
            echo "$user created successfully"
        fi
    fi
done < user.txt
```

```bash
# 如果网络环境正常，就一直send，如果网络环境异常，就跳出循环、
ip=10.80.5.25
while ping -c1 -w1 $ip &>/dev/null
do
	# send message
	sleep 1
done
echo "$ip is down"
```

### 3.3.3 util循环

条件为假，执行循环。条件为真跳出循环，和while正好相反

```bash
#do循环
until [ condition ]
do
	程序段
done
```

```bash
# 如果网络环境异常，就一直send，如果网络环境正常，就跳出循环、
ip=10.80.5.25
util ping -c1 -w1 $ip &>/dev/null
do
	# send message
	sleep 1
done
echo "$ip is up"
```

### 3.3.4 循环汇总

```bash
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
```

### 3.3.5 并发控制（控制并发数量）

在上面的循环程序中，每执行一次循环体（后台执行），就会开启一个新的子进程，当循环体被执行了1000次，如果循环体内的执行过程较长，极有可能就会累积1000个子进程。

就像下面这个批量后台创建1000个用户

```bash
for i in `seq -w 1000`
do
	{
        user="u$i"
        useradd user
        echo 123456 | passwd --stdin $user &>/dev/null
        if [ $? -eq 0 ]; then
            echo "$user is created successfully"
        fi
	}&
done
```

并发控制（控制并发数量），需要一下两个基础知识点：

1. 文件描述符（文件句柄）
   - 查看当前进程中，打开的文件描述符列表：`ll /proc/$$/fd`，fd——file description，$$——表示当前进程
   - 打开file1文件并且指定文件描述符序号：`exec 6<> /file1`
   - 关闭file1文件并且释放文件描述符：`exec 6>&-`
   - 其实我们修改文件就是在修改文件描述符
   - 当一个文件FD未被释放，删除源文件也不会影响FD，并且还可以通过fd恢复源文件
2. 匿名管道和命名管道
   - 管道对应的就是一个管道文件
   - 匿名管道，就像一般的管道命令，这个文件没有名字，暗送，eg：`rpm -qa | grep bash`
     - 匿名管道是由pipe函数创建 并打开的
     - 在同一个终端上
   - 命名管道，通过创建具体/path/file1文件来实现，eg：`mkfifo /tmp/tmpfifo`，
     - 命名管道是由mkfifo函数创建 的 
     - 可以实现不同终端之间的通信，一个端子向fifo文件里写数据，一个端子从fifo文件读数据。
   - 管道里的数据，一旦看了或用了就没了， 并且是fifo（first input first output）

#### 并发控制实例

```bash
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
```

### 3.3.6 shell内置命令

冒号**:**：空语句，永真，占位符

break：结束当前循环，跳出本层循环

continue：忽略本次循环剩余代码，直接进行下一次循环

shift：使位置参数（函数和脚本通用）向左移动，默认移动一位，shift 2就移两位

```bash
while [ $# -ne 0 ]
do
	useradd $1 # 添加用户，./script.sh alice qqq kkk hhh
	# let sum+=$1 # 所有参数求和，./script.sh 1 2 3 4
	shift 1 # 执行后，命令后面的第一参数会被踢出参数队列，而执行前的第二个参数被推向了参数队列的首位
done
```



# 4 数组

普通数组：只能用整数作为数组索引

关联数组：可以使用字符串作为数组索引，类似于java的类

## 4.1 数组定义

```bash
# 1.数组声明
# shell默认能识别普通数组，而不能识别关联数组
declare -aA [数组名]
# a，声明普通数组
# A，声明关联数组，在定义关联数组之前，一定要声明，否则引用数组元素将会出错
# 如果没有数组名，将会打印出当前环境下的所有当前类型的数组
declare -a # 查看当前环境，所有的普通数组
declare -A # 查看当前环境，所有的关联数组

# 2.普通数组定义
# 一次赋一个值
fruit[0]=apple
fruit[1]=pear
# 一次赋多个值
books=(linux shell awk openstack docker) 
arr1=(`cat /etc/passwd`) # 期望该文件的每一行作为一个数组元素，但记住默认是以空格为分割对象，需要修改IFS
arr2=(`ls /var/ftp/shell/for*`)
arr3=(qqq kkk "hhh ni") # 数组只有三个元素
arr4=($red $blue $orange)
arr5=(1 2 3 "linux shell" [20]=puppet) # 可以跳过中间索引值

# 3.关联数组定义
# 先声明
declare -A info1
# 一次赋一个值
sex=([m]=1)
sex+=([f]=1) # echo ${!sex[@]} m f
# 一次赋多个值
info1=([name]=qqq [sex]=male [age]=30 [height]=170)

# 4.访问数组
echo ${fruit[0]}	# 访问数组第一个元素
echo ${fruit[@]} # 访问数组所有元素，等价于echo ${fruit[*]}
echo ${#fruit[@]} # 统计数组元素个数
echo ${!fruit[@]} # 获取所有数组元素的索引（包括关联数组的索引）
echo ${fruit[@]:1} # 从数组索引1开始切片，关联数组不能切片
echo ${fruit[@]:1:2} # 从数组索引1开始切2个数组元素

```

## 4.2 数组赋值与遍历

 ```bash
 #!/usr/bin/bash
 while read line
 do
     hosts[i++]=$line
 done < /etc/hosts
 
 echo "hosts first: ${hosts{0}}"
 echo
 
 # ${!hosts[@]}返回数组的所有索引，以空格分割
 for i in ${!hosts[@]}
 do
     echo "${i} : ${hosts[i]}"
 done
 ```

## 4.3 应用案例

### 4.3.1 统计性别

```bash
#!/usr/bin/bash
# sex.txt
# qqq f
# kkk m
# hhh x
# rrr f
# yyy f
# xxx m
# ttt x

declare -A sex

while read line
do
    type=`echo $line | awk '{print:$2}'`
    # 这里可以在关联数组里面主动加键名
    let sex[$type]++
done < sex.txt

for i in ${!sex[@]}
do
    echo "$i : ${sex[$i]}"
done


uuser=(HwHiAiUser, HwDmUser, HwBaseUser, HwSysUser)

for ind in `seq 0 $((${#uuser[@]}-1))`
do
    echo "${ind} : ${uuser[ind]}"
    sleep 1
done
```

### 4.3.2 统计shell类型的数量

```bash
#!/usr/bin/bash
declare -A shells

while read line
do
    type=`echo $line | awk -F ":" '{print $NF}'`
    let shells[$type]++
done </etc/passwd

for ind in `seq 1 ${#shells[@]}`
do
    echo "${ind} : ${shells[ind]}"
    sleep 1
done
```

### 4.3.3 统计tcp链接状态的数量

```bash
declare -A status
type=`ss -an | grep :80 | awk '{print $2}'`

for i in $type
do
    let status[$i]++
    let shells[$type]++
done

for j in ${!status[@]}
do
    echo "$j : ${status[$j]}"
done
```

`watch -n2 ./count_tcpconn_status.sh`，每隔两秒统计一次

当人也可以用while和sleep周期性统计

```bash
....
while :
do
unset status
	....
	sleep 2
	clear
done
```

# 5 函数

完成特定功能的代码块，使代码模块化，便于复用代码

## 5.1 函数定义

```bash
# 1.方法1
fun_name() {
	# 函数体
	# 函数体里的$1，代表函数的第一个参数
	# 函数的返回值是，函数体内最后一条命令的返回码
	# 如果采用return的方式，shell最大的返回值（码）是255，大于这个值将会得到一个错误的值，像浮点数等其他值就无法返回了
	# 在函数内部推荐使用local声明的变量，因为这样不会影响shell的主体变量内容
}
# 2.方法2
function fun_name {
	# 函数体
	# $1,$2,$3,$@,$#
	# return
}

# 3.使用
# 函数必须先定义再调用，否则报错
# 函数如果在其他shell脚本中定义，必须 “先” 在父shell中执行其他脚本(. script.sh)，然后再使用该函数
# 函数传参通过如命令的方式，直接续在函数名之后
fun_name param1 param2
```

```bash
#/bin/env bash
# 这样的shebang，可以自动去查找bash的位置

levelmultiple() {
    res=1
    # 函数里的$1是函数的位置参数，$1是函数的第一个参数
    for ((i=1;i<=$1;i++)) #类c写法
    do
        # res=$((res * i))
        let res*=i
    done
    echo "$1 的阶乘：$res"
    # return方式的返回值最大是255，
    return 200
}

# 最外层的$1是脚本的位置参数，$1是脚本的第一个位置参数
levelmultiple $1
echo $? # 这里将会打印200
```

### 函数体内的变量

函数体内的变量

- 不加local，那么函数外也可以获取到此变量的值，该变量属于全局变量
- 加local，那么函数外不可获取此变量的值，该变量属于局部变量

```bash
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
# 命令替换，相当于在子shell中执行，执行后，c就消失了
result=`fun3;echo "子shellc：$c"`
echo "c：$c" # 无法获取
```

## 5.2 函数返回任意值

可以通过echo和变量的方式获取函数的返回值

```bash
#!/usr/bin/bash
doublenum() {
    read -p 'please input a number： ' num
    # echo 'computing...'，注意此时函数里只能有一个echo,或者标准输出只能有一个
    echo $((2 * num))
}

result=`doublenum`
echo "doublenum return value: $result"
```

## 5.3 函数的输入和输出为数组

```bash
# 1.函数的输入为数组
#!/usr/bin/bash
num1=(1 2 3)
# 以空格做分隔显示所有数组元素
echo ${num1[@]}

# 定义函数
arr_fun() {
    echo $*
    # 如果函数里的变量
    # 不加local，那么函数外也可以获取到此变量的值，该变量属于全局变量
    # 加local，那么函数外不可获取此变量的值，该变量属于局部变量
    local res=1
    
    # 如果用数组元素作为实参，那么函数体内通过$*,$@接收
    for i in $*
    # for i in $@
    do
    	echo $i
        res=$[res * i]
    done
}

# 函数调用，将数组的所有元素都做实参数
arr_fun ${num1[@]}
```

```bash
#2.函数的输入输出都为数组
num1=(1 2 3)
arr_fun() {
    # $*本来就是参数的空格相隔的参数，用括号括起来就是定义数组
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
```

# 6 文本处理

元字符是这样一类字符，他们表达的是不同于字面本身的含义

shell元字符（也称通配符）：由shell来解析（在1.5节有描述）

正则表达式元字符：由各种执行模式匹配操作的程序来解析

## 6.1[正则基础](https://www.zhihu.com/question/48219401/answer/742444326)

**正则表达式的通项： /pattern/flags**

| 符号\|组别       | 解释                                                         |
| ---------------- | ------------------------------------------------------------ |
|                  |                                                              |
| **合法字符**     |                                                              |
|                  |                                                              |
| **"  x  "**      | x代表任意合法字符                                            |
| **"  \uhhhh  "** | 十六进制所代表的Unicode字符                                  |
| **"  \t  "**     | 制表符                                                       |
| **"  \n  "**     | 换行符                                                       |
| **"  \r  "**     | 回车符                                                       |
| **"  \f  "**     | 换页符                                                       |
| **"  \cx  "**    | x对应的控制符，                                              |
|                  |                                                              |
| **预定义字符**   |                                                              |
|                  |                                                              |
| **"  .  "**      | 匹配任意**一个**字符                                         |
| **"  \b  "**     | 匹配词边界                                                   |
| **"  \d  "**     | 匹配所有数字                                                 |
| **"  \D  "**     | 匹配所有非数字                                               |
| **"  \s  "**     | 匹配所有空白符，包括空格，制表符，回车符，换页符，换行符等   |
| **"  \S  "**     | 匹配所有非空白符                                             |
| **"  \w  "**     | 匹配所有单词字符，包括0-9，26个字母和下划线                  |
| **"  \W  "**     | 匹配所有非单词字符                                           |
|                  |                                                              |
| **特殊字符**     |                                                              |
|                  |                                                              |
| **"   ^  "**     | 匹配一行的起始，eg：" ^a "代表匹配所有以字母a开头的字符串    |
| **"  $  "**      | 匹配一行的结尾，eg：" ^$ "代表匹配空行                       |
| **"  [  ]  "**   | 匹配字符集里的多个字符，存在集里的字符即匹配，eg：[0-9c]匹配0到9或c中的任意多个字符 |
| **"  (  )  "**   | 匹配子表达式                                                 |
|                  |                                                              |
| **重复限定符**   |                                                              |
|                  |                                                              |
| **"  {  }  "**   | 指定其前面子表达式可以出现的次数，<br />**{ n }——重复n次、{n , }——重复大于等于n次、{ n , m }——重复n到m次**<br />eg：(a){3}，a连续出现三次， |
| **"  *  "**      | 指定其前面子表达式可以出现0或多次                            |
| **"  +  "**      | 指定其前面子表达式可以出现1或多次                            |
| **"  ?  "**      | 指定其前面子表达式可以出现0或1次                             |
|                  |                                                              |
| **"  \|  "**     | 指定竖线两侧的两项中任选一项匹配                             |
| **"  \  "**      | 转义字符                                                     |
|                  |                                                              |
| **边界匹配符**   |                                                              |
|                  |                                                              |
| **"  \b  "**     | 单词的边界，即只能匹配到单词前后的空白                       |
| **"   \B  "**    | 非单词的边界                                                 |
| **"   \A  "**    | 只匹配字符串的开头                                           |
| **"   \Z  "**    | 只匹配字符串的结尾                                           |

<b>方括号表达式</b>

| 用途            | 解释                                     |
| --------------- | ---------------------------------------- |
| **表示枚举**    | eg：[abc]                                |
| **表示范围**    | eg:[a-z]                                 |
| **表示求否：^** | eg:[**^**abc]                            |
|                 |                                          |
| **匹配模式**    |                                          |
|                 |                                          |
| **"  i  "**     | ignorCase忽略大小写                      |
| **"  g  "**     | globle进行全局匹配，指匹配到目标串的结尾 |
| **"  m  "**     | mutiple允许多行匹配                      |

### 6.1.1断言匹配

断言只是匹配位置，也就是说，匹配结果里是不会返回断言本身，返回的是断言前后的符合另一个正则表达式的内容

1. 正向先行断言：

   - 找到pattern的位置，然后找该位置之前的prePattern内容作为匹配结果返回

   - ```js
     //格式
     prePattern(?=pattern)
     ```

   - 

2. 正向后行断言

   - 找到pattern的位置，然后找到该位置之后的affixPattern内容作为匹配结果返回

   - ```js
     //格式
     affixPattern(?<=pattern)
     ```

   - 

3. 负向前行断言

   - 负向的意思是对pattern进行取非操作，然后找到非pattern的位置，再将该位置之前的prePattern内容作为匹配结果返回

   - ```js
     //格式
     prePattern(?!pattern)
     ```

   - 

4. 负向后行断言

   - 负向的意思是对pattern进行取非操作，然后找到非pattern的位置，再将该位置之后的affixPattern内容作为匹配结果返回

   - ```js
     //格式
     affixPattern(?<!pattern)
     ```

   - 

### 6.1.2 捕获

单纯说到捕获，他的意思是匹配表达式，但捕获通常和分组联系在一起，也就是“捕获组”。

一个正则表达式里有多个子表达式（用括号括起来的表达式，eg：（[a-z]|\d），这些子表达式就作为一个正则表达式的分组。

匹配子表达式的内容，把匹配结果保存到内存中，用数字编号或显示命名的组里，以深度优先进行编号，之后可以通过序号或名称来使用这些匹配结果。

1. 数字编号捕获组
   - 从表达式左侧开始，每出现一个左括号和它对应的右括号之间的内容为一个分组，在分组中，第0组为整个表达式，第1组开始为分组。
   - 格式：(pattern)，，eg：(0\d{2})-(\d{8})，第0组为整个表达式，第一分组为(0\d{2})，第二分组为(\d{8})
2. 命名编号捕获组
   - 分组的命名由表达式中的name指定
   - 格式：(?\<name>pattern)，eg：(?\<quhao>\0\d{2})-(?\<haoma>\d{8})
3. 非捕获组：
   - 用来标识不需要捕获的分组，就是可以根据你的需要来捕获分组。
   - 格式：(?:pattern)，eg：(?:\0\d{2})-(?\<haoma>\d{8})

### 6.1.3 反向引用

通常捕获组和反向引用是组合使用的。

举个例子：假如你想匹配到aa，bb，cc，11这样的内容。

- 1）匹配到一个字母
- 2）匹配第下一个字母，检查是否和上一个字母是否一样
- 3）如果一样，则匹配成功，否则失败

这里的思路2中匹配下一个字母时，需要用到上一个字母，那怎么记住上一个字母呢？？？
这下子捕获就有用处啦，我们**可以利用捕获把上一个匹配成功的内容（引用捕获分组的内容）用来作为本次匹配的条件**

- 1）匹配到一个字母，\w
- 2）匹配第下一个字母，检查是否和上一个字母是否一样，(\w)\1，\1的意思是引用了(\w)捕获组的内容，这样就会匹配到和(\w)一样的字母了
- 3）如果一样，则匹配成功，否则失败

根据捕获组的命名规则，反向引用可分为：

1. 数字编号组反向引用：\k 或 \number
2. 命名编号组反向引用：\k 或者 \\'name'

### 6.1.4 贪婪

贪婪与懒惰

#### 贪婪

当正则表达式中**包含能接受重复的限定符**时，通常的行为是（在使整个表达式能得到匹配的前提下）匹配尽可能多的字符，这匹配方式叫做贪婪匹配。

```js
let str="61762828 176 2991 44 871";
let pattern=/\d{3,6}/
//617628  176  2991  871
```

按照表达式来，在匹配到617这三个字符已经成功匹配到了一个，但是它并不满足，所以它要继续匹配，看看它能不能匹配到更长的结果，结果第一个匹配的结果为617628。

这就是贪婪。

#### 懒惰

懒惰匹配：当正则表达式中包含能接受重复的限定符时，通常的行为是（在使整个表达式能得到匹配的前提下）匹配尽可能少的字符，这匹配方式叫做懒惰匹配。
特性：从左到右，从字符串的最左边开始匹配，每次试图不读入字符匹配，匹配成功，则完成匹配，否则读入一个字符再匹配，依此循环（读入字符、匹配）直到匹配成功或者把字符串的字符匹配完为止。

在重复限定符后面加？号，就构成了懒惰匹配。

```js
*?//任意次，但尽可能少
+?//一次或多次，但尽可能少
??//0次或一次，但尽可能少
{n,m}?//n次到m次，但尽可能少
{n,}?//n次以上，但尽可能少
```

## 6.2 shell正则细节

shell元字符：`^ $ . * [] [-] [^] \ \< \> \( \) \{ \} `

在shell中元字符需要加**`\`**才能起作用

- 词首（<）词尾（>）定位符，使用时`\<   \>`
- 子表达式`( expression )`，使用时`\( expression \)`
- 重复限定符`{ m,n }`，使用时`\{ m,n \}`
- `+ |`，在使用时`\+ \|`

shell扩展元字符：`+ ?  |  ()  {}`

## 6.3 grep族

对行做操作，在文件或指定输入（管道，标准输入）中查找指定的正则表达式，并打印所有包含该表达式的行。

1. grep，
2. egrep，扩展的grep，支持更多的元字符，推荐使用egrep
3. fgrep，fixed grep，按字面意思解释所有字符



`grep [选项] "pattern" file1 file2...`

- pattern，记得要加引号，因为如果pattern中包含空格，那么空格后的内容，容易被当做file名
- 返回值`$?`
  - 0，可以找到包含匹配项的行
  - 1，不可以找到包含匹配项的行
  - 2，找不到指定的文件
- grep 能使用基本元字符，如果要使用扩展元字符：`grep -E`，**推荐使用egrep**
- grep选项
  - -i，--ignore-case，忽略大小写
  - -q，--quit，--silent，打印到屏幕
  - -v，--invert-match，反向匹配，只显示不匹配的行
  - -R，-r，--recursive，针对目录，递归查找
  - --color，centos7开始默认支持对匹配项加颜色标注
  - -o，只返回匹配的内容，而不返回整行
  - -B，--before-context，eg：grep -B2，会连带显示匹配行的前两行
  - -A，--after-context，-A2，会连带显示匹配行的前2行
  - -C，--context，-C2，会连带显示匹配行的前后两行
  - -l，只列出含有匹配内容的文件名
  - -n，--line-number，连带显示匹配行的行号
- 

```bash
# 如果想要查找某个命令帮助内容中某个选项的相关内容，
#grep 后面的-u不能直接写，而需要一个\去转义-，否则它会认为-u是grep的选项，而不是pattern
useradd --help | grep '\-u'
# 简单的ip地址匹配
([0-9]{1,3}\.){3}[0-9]{1,3}
([0-9]{1,3}).\1.\1.\1	# 涉及捕获组命名规则
```

## 6.4 流编辑器sed

vim是一种交互式的编辑器，而sed是一种**在线的，非交互式**的编辑器。

它一次处理一行内容。

处理时，把当前处理的行存储在临时缓存区中，称为“模式空间”

接着，用sed命令处理缓存区中的内容，处理完成后，把缓存区的内容送往stdout。接着处理下一行，直到文件末尾。

该操作不会改变源文件的实际内容，除非将输出内容重定向到其他文件中进行存储。

sed主要用来自动编辑一个或多个文件，简化对文件的反复操作，编写转化程序等。

<img src="./legend/sed的工作流程.jpeg" style="zoom:67%;" />

sed 命令格式

- 命令的方式：`sed [option] 'command' file(s) `，这里的命令是sed特有的命令，后面会说到
- 脚本的方式：`sed [option] -f scipts_file file(s)`

sed命令只有在有语法错误的时候返回非0，其余状态都返回0，所以在sed里，通常不把$?作为命令是否执行成功的依据。

sed同grep'一样支持正则表达式，默认情况下，可以使用基本元字符，**pattern需要写在双斜线（可以替换为其他字符eg：#@$，如果在查找模式下，需要将上述的第一个进行转义，如果是在替换模式可以不需要转义）之间。**

### 6.3.1 sed选项

1. -r，支持使用扩展元字符
2. -n，--quit，--silent
3. -f
4. i，in place，就地编辑，会修改源文件
5. e，多重编辑，顺承编辑操作，将文件做了一次sed之后（所有行都操作过了），在此基础上，再做一次sed

```bash
# s命令，substitute替换，将第一个正则/root/匹配到的内容，替换成alice
sed -r 's/root/alice/' /etc/passwd
sed -r 's@root@alice@' /etc/passwd	#正则表达式双斜杠符号可以直接替换为@

# d命令，delete删除，删除含有匹配内容的行
sed -r '/root/d' /etc/passwd
sed -r '\@root@d' /etc/passwd	#在查找模式中，正则表达式双斜杠符号必须在第一个斜杠时转义，然后可替换为@

sed -e '1,3d' -e 's/home/admin/' /etc/passwd
	# 第一个-e，将第一行到第三行删除
	# 得到没有1，2，3行的内容后
	# 再进行第二个-e的替换操作
# 等价于
sed -e '1,3d;s/home/admin/' /etc/passwd
```

### 6.4.2 定址

地址用于决定对那些行进行编辑处理

地址的形式可以是数字，正则表达式或二者的结合

如果没有指定地址，sed将会处理文件中的所有行

**格式：`[address]command`**

```bash
# 数字定址
sed -r '3d' /etc/passwd # 删除第三行
sed -r '3,5d' /etc/passwd # 删除第三行到第五行
sed -r '3,$d' /etc/passwd  # 第三行到最后一行

# 正则定址，adress——[/root/]
sed -r '/root/d' /etc/passwd
sed -r '/^#/d' /file.conf # 删除配置文件中的#注释行

# 混合定址，adress——[/root/,5]，command——d
sed -r '/root/,5d' /etc/passwd # 删除从root行开始，到第五行

# 特殊符号定址
sed -r '/^bin/,+5d' /etc/passwd # 删除从/^bin/开始，再删后面5行
sed -r '/root/!d' /etc/passwd # 反向删除，除了匹配行，其余都删
sed -r '1~2d' /etc/passwd # 删除所有奇数行
sed -r '0~2d' /etc/passwd # 删除所有偶数行

sed -r 's/(.*)/#\1/' /etc/passwd	# 在所有行前加#，首先匹配所有行(.*),然后替换成#\1，\1是匹配的内容
#等价于
sed -r 's/.*/#&/' /etc/passwd		# &在替换的内容里面表示前面匹配的内容，和\1是一致的
```

### 6.4.3 sed命令

#### 命令组合

当多个命令都作用于同一地址行，就可使用：

`address{command1;command2;...commandn}`

```bash
sed -r '3{d;h}' /etc/passwd
```

和多重编辑不同，这里的两个命令都针对于第三行，而不是多重编辑的两次sed操作。

#### 暂存空间

在模式空间之外还有一个暂存空间（保持空间），用于腾挪内容之用

模式空间的定址行可以存到暂存空间，在需要的时候再将暂存空间的内容放到模式空间，两个空间的内容可以交互。

暂存空间初始时有一个空行

#### 命令解析

```bash
# d，delete，删除定址的行

# s，substitute，替换
sed -r 's/root/alice/g' /etc/passwd #g一行中所有匹配到的内容都需要替换
sed -r 's/root/alice/gi' /etc/passwd #i忽略大小写
sed -r 's/(.)(.)(.*)/\1\2yyy\3/' /etc/passwd # 每行的第二个字母后加yyy
sed -r 's@root@alice@' /etc/passwd	# 格式分割符/可以直接替换为@

# r，读入指定的文件内容到定址行后面，指定的文件与r间可以没有空格，这里有空格只是为了阅读更友好
sed -r '/lp/r /etc/hosts' /etc/passwd # 将/etc/hosts文件的内容，读入到定址行的后面

# w， 将定址行的内容写入到其他文件中
sed -r '/lp/w /tmp/1.txt' /etc/passwd

# a，在定址行后面插入指定内容
sed -r '2a\1111' /etc/passwd
sed -r '2a\1111
2222\
3333' /etc/passwd  # 插入多行内容
# i，在定址行前面插入指定内容
# c，将定址行（整行）替换为指定内容

# n，操作定址行的下一行，通常配合组合命令使用
sed -r '/adm/{n;s/sbin/uuu}' /etc/passwd # 定址行为包含/adm/内容的行,n——它的下一行，做替换操作
sed -r '/adm/{n;n;n;s/sbin/uuu}' /etc/passwd # n可以用多次，下下下一行

# h，H和g，G 暂存和取用命令
# h，把模式空间里的内容复制后，覆盖暂存空间当前的内容（覆盖的方式）
# H，把模式空间里的内容复制后，追加到暂存空间当前内容的后面（追加的方式）
# g，把暂存空间的内容复制后，覆盖模式空间当前的内容
# G，把暂存空间的内容复制后，追加到模式空间当前的内容的后面

sed -r '1h;$G' /etc/passwd 
	# 复制第一行内容，覆盖暂存空间（初始的空行被覆盖掉）；
	# 再匹配到最后一行（$）内容的时候，再将暂存空间的内容（原第一行的内容）追加的最后一行的后面
sed -r '1{h;d};$G' /etc/passwd
	# 剪切第一行内容，并追加到最后一行之后
sed -r '1h;2,$g'  /etc/passwd
	# 将第二行到最后一行，都覆盖为第一行的内容
sed -r '1h;2,3H;$G' /etc/passwd
	# 将1,2,3行的内容追加到最后一行之后

# x，交换暂存空间与模式空间的内容
sed -r '4h;5x;6G' /etc/passwd
	#...4465...
	
# !，反向选择
sed -r '3!d' /etc/passwd
	# 除了第三行以外的内容都删除

```

#### sed中使用变量

```bash
# 在最后一行后，追加$var1的内容
# 法一：有变量的地方用双引号
sed -ri '$a'"$var1" /file
# 法二：不用引号
sed -ri $a$var1 /file
# 法三：双引号下转义：最后一行$
sed -ri "\$a$var1" /file
```

### 6.4.4 操作实例

```bash
# 删除文件中的#注释行
sed -ri '/^[ \t]*#/d' /file
# 删除文件中的//注释行
sed -ri '\@^[ \t]*//@d' /file
# 删除文件中的空行
sed -ri '/^[ \t]*$/d' /file
# 删除空行和#注释行
sed -ri '/^[ \t]*$ | ^[ \t]*#/d' /file
sed -ri '/^[ \t]*[#|$]/d' /file

# 给文件行添加注释
sed -r 's/(.*)/#\1/' /file
sed -r 's/.*/#&/' /file
# 将行首的0个或多个空格、tab、#，换成一个#
sed -r 's/^[ \t#]**/#/' /file
```

## 6.5 awk

awk是一种编程语言，它能提供一个类编程环境来修改额重新组织数据。它也是逐行扫描数据

数据可以来自标准注入，文件，管道。

特点：

- 定义变量来保存数据
- 使用算术和字符串操作符来处理数据
- 使用结构化编程概念（选择结构，循环结构）为数据增加处理逻辑
- 提取数据文件中的数据元素，将其重新排列或格式化，生成格式化报告

gawk程序是Unix中原始awk程序的GNU版本。所有linux发行版都没有默认安装gawk，需要自行安装。它提供了一种编程语言而不只是编辑器命令。

awk命令格式

- 命令方式：`awk [option] 'command' files`
- 脚本方式：`awk [option] -f awk_scirpt_file files`

### 6.5.1 awk工作流程

`awk 'BEGIN{FS=":"}{print $1,$2}'`

1. awk使用一行作为输入，并将这一行赋给内部变量$0，**每一行可称为一个记录**，以换行符结束
2. 行被内部变量FS（字段分隔符，默认空格 or tab）分隔成字段，每个字段被依序存储在内部变量$1，$2，...$n，最多可以到100。
3. 在处理时，原命令的逗号被映射为OFS（output FS），打印出$1,$2
4. 一行处理后，换另一行，重复上述动作，直到所有行处理完毕

### 6.5.2 awk选项

- F，定义每行输入数据的字段分割符，默认是空格和tab
- f，指定awk命令脚本文件
- 

### 6.5.3 awk内部变量

1. $0：当前处理行的内容
2. NR：当前行被处理的次序号（多个文件输入时，会依次递增）
3. FNR：当前行被处理的次序号（多个文件输入时，一旦开始处理下一个文件，次序号会重新归1）
4. NF：当前行一共有多少个字段，$NF是最后一个字段的内容
5. FS：输入内容的分隔符，默认空格和tab
6. OFS：输出内容的分隔符。默认空格
7. RS：记录（行）分隔符。默认是换行符
8. ORS：输出记录（行）的分隔符。默认是换行符

```bash
# 一行按照多种分隔符分隔字段，并且多个空格，多个：，多个tab都只算一个分隔符（符合正则语法）
awk -F'[ :\t]+' '{print $1,$2,$3}' #这里按照空格，冒号，tab同时分
```

### 6.5.4 awk命令

任何awk命令语句都是由**模式和动作**组成，模式里包括两个特殊模式（BEGIN，END），动作则由花括号括起来的语句

模式用于匹配数据行，而动作则是对匹配行进行操作

**`BEGIN{}		pattern{}		END{}`**

1. **BEGIN模式**

   - 发生在读文件之前（读第一行数据之前）

   - 可以不要files参数，而另外两个过程需要files参数
   - 通常用于定义一些变量，例如FS（字段分隔符），OFS（输出字符分隔符）
   - 可以省略该模式

2. **其他pattern模式**：处理行数据，**其他模式可以是任何条件语句或复合语句或正则**，省略则表示匹配所有行

3. **END模式**：所有行处理结束之后，可以省略该模式

三个过程之间可以没有分割（空格），一般分割是为了阅读友好

模式之间在书写上也没有次序之分，谁先谁后都一样，按BEGIN，其他，END是为了阅读友好

```bash
awk 'BEGIN{print 1/2} {print "ok"} END{print "------"}' /etc/hosts	
0.5		# 在读文件之前打印了一个1/2
ok		# 在处理每行的时候，输出ok
ok
ok
ok
------	# 在处理完所有行之后，输出-------
# 字符串需要用双引号括起来，否则无法打印

# BEGIN定义变量
awk 'BEGIN{FS=":";OFS="-----"} {print $1,$2}' /etc/passwd
root-----x
bin-----x
daemon-----x
adm-----x

```

#### 函数

一般输出和格式化输出

```bash
# 将年月换行输出
date | awk '{print "month: " $2 "\nYear: " $NF}' # 只要想原样输出的都应放置在双引号之中

# 定宽输出，
awk -F: '{printf "%-15s %-10s %-15s\n",$1,$2,$3}' /etc/passwd # 第一个字段15个字符，第二个字段10个字符
awk -F: '{printf "|%-15s |%-10s |%-15s|\n",$1,$2,$3}'
# %s——字符类型，%d——数值类型，%f——浮点类型，-表示左对齐（默认右对齐）
# printf默认不会在行尾自动换行，需要自行添加换行符\n或其他
```

### 6.5.5 命令模式

```bash
# 正则表达式
# 匹配整行（记录）
awk '/^root/' /etc/passwd # 把以root开头的行打印出
awk '!/root/' /etc/passwd # 把不是以root开头的行打印出
# 匹配字段，等于不等于（~ !~）
awk -F: '$1 ~ /root/' /etc/passwd # 第一个字段内容等于root，就打印
awk -F: '$1 !~ /root/' /etc/passwd # 第一个字段内容不等于root，就打印

# 比较表达式
# 利用关系运算符来比较数字和字符串(字符串需要用双引号括起来)
# 关系运算符，< > <= >= == != 
awk -F: '$3 == 0' /etc/passwd
awk -F: '$7 == "/bin/bash"' /etc/passwd
awk -F: '$3>300{print $0}' /etc/passwd
# 等价于
awk -F: '{if($3>300) print $0}' /etc/passwd	# 把比较运算放在行处理里面
awk -F: '{if($3>300) {print $0} }' /etc/passwd

awk -F: '{if($3>300) {print $0} else {print $1} }' /etc/passwd

# 算术运算，+ - * / % ^指数
awk -F: '$3*10 > 500' /etc/passwd
awk -F: '{if($3*10 > 500){print $0}}' /etc/passwd

# 逻辑与复合模式
# && 与，|| 或，!非
awk -F: '!($1~/root/ || $3 <= 15)' /etc/passwd

# 范围
awk -F: '/Tom/,/susan/' /etc/passwd # 从TOm行到Susan行

# 三目运算符
awk '{print ($7>4 ? "high":"low"$7)}' datafile
# 赋值运算
awk '$3=="chris"{$3="chris你好";print $0}' datafile
# 短加运算
awk '{$7%=3;print $7}' datafile
```



### 6.5.5 命令动作——awk脚本编程

注意：这里只写出动作部分，没有写全命令

#### 程序结构

```bash
# 1.条件结构
{ if(expression){sentence1; sentence2;} }
{ if(expression){sentence1; sentence2;} else {sentence1; sentence2;} }
{ if(expression){sentence1; sentence2;} else if(expression) {sentence1; sentence2;} else {sentence1; sentence2;} }

# 统计符合$3字段大于0小于100的行的行数,
{ if($3>0 && $3<100){ count++ }} END{print count}
awk -F: '{if($3==0){i++} else{count++} } END {print "管理员个数："i; print "系统用户个数："count; }'

# 2.循环结构
{ i=1; while(i<=NF){print $i;i++} }	
	# 打印每行的所有字段
	# 每行都要执行这个循环，记得i复位，否则后面的行初始拿到的i的值就是之前行累加过的值
	# 循环记得要有出口，否则死循环
{ for(i=1;i<NF;i++){print $i} }
```

#### awk数组

在awk中使用数组不需要提前声明，**本身支持关联数组类型**

```bash
# 数组生成
awk -F: '{i=0;username[i++]=$1} END{print username[0]}' /etc/passwd
# 等价
awk -F: '{username[i++]=$1} END{print username[0]}' /etc/passwd # 数组的开始索引为0
awk -F: '{username[++i]=$1} END{print username[0]}' /etc/passwd # 数组的开始索引为1

# 数组遍历
# 元素个数遍历（不推荐），因为容易搞不清起始索引,导致后面的for你不知道用<=还是小于
awk -F: '{username[i++]=$1} END{ for(j=0;j<i;j++){print i,username[j]} }
awk -F: '{username[i++]=$1} END{ for(j=0;j<=i;j++){print i,username[j]} }
# 按索引遍历
awk -F: '{username[i++]=$1} END{ for(i in username){print i,username[i]} }'  /etc/passwd

# 统计shell的类型，这里包含关联数组
awk -F: '{shells[$NF]++} END{for(i in shells){ print i,shells[i]} }'  /etc/passwd
# 统计tcp连接状态
netstat -ant | grep ':80' | awk '{status[$NF]++} END{for(i in status){print i,status[i]} }'
```

#### awk使用外部变量

```bash
var=bash
# 法一：command 用双引号，变量用双引号，再用反斜杠转义
echo "unix scripts" | awk "gsub(/unix/,\"$var\")"
# 法二：command 用单引号，变量用双引号套单引号"'"$var"'"，
echo "unix scripts" | awk 'gsub(/unix/,"'"$var"'")'
# 法三：command 用单引号，变量用三个单引号,不能在函数中使用
df -h | awk '{if(int($5) > '''$var'''){print $6":"$5}}'
# 法四：awk -v
echo "unix scripts" | awk -v var="bash" 'gsub(/unix/,var)'
```

### 6.5.6 函数

```bash
# 一般输出print 
# 将年月换行输出
date | awk '{print "month: " $2 "\nYear: " $NF}' # 只要想原样输出的都应放置在双引号之中

# 格式化输出printf
# 定宽输出，
awk -F: '{printf "%-15s %-10s %-15s\n",$1,$2,$3}' /etc/passwd # 第一个字段15个字符，第二个字段10个字符
awk -F: '{printf "|%-15s |%-10s |%-15s|\n",$1,$2,$3}'
# %s——字符类型，%d——数值类型，%f——浮点类型，-表示左对齐（默认右对齐）
# printf默认不会在行尾自动换行，需要自行添加换行符\n或其他

# 计算变量的长度length
awk -F: 'length($1)==4{count++;print $1} END{print "count is "count}' /etc/passwd # 统计用户名长度为4的个数

# 查找替换sub，全量查找替换gsub
echo "unix script" | awk 'gsub(/unix/,bash)'

# 转整数int，int(20%)=>20
```

### 

# 常用

1. [basename](https://blog.csdn.net/weixin_40734030/article/details/122674137)：用于打印目录或者文件的基本名称，

   - ```bash
      basename ./scripts/预定义变量.sh
      # 打印出
      预定义变量.sh
     ```

   - 类似的还有dirname，打印出目录或路径的名字

   

2. [command](https://blog.51cto.com/dlican/5097063)：调用指定的指令并执行，命令执行时不查询shell函数。command命令只能够执行shell内部的命令。

   - **command [-pVv] command1 [参数 ...]**

     - p，类似于type命令，用来查询，command1是不是可执行的内部命令

     ```bash
     #!/usr/bin/bash
     
     # 动态执行命令
     read -p "input command: " command1
     command $command1
     
     # 根据命令的有无，去安装某些软件
     read -p "input command: " command2
     if command -v $command2 &>/dev/null; then
     	echo "$command2 is already existed”
     else
     	echo "yum install"
     	# yum -y install 
     fi
     ```

   - 

3. 冒号(:)的作用

   - 空命令

   - 参数扩展
   - [重定向](https://so.csdn.net/so/search?q=重定向&spm=1001.2101.3001.7020)
   - 当注释使用

4. whoami

   - 用于显示自身用户名称。
   - 相当于执行"id -un"指令

5. [wait](https://www.jb51.net/article/272457.htm)

   - 它**等待**后台运行的进程完成并返回退出状态。与等待指定时间的sleep 命令不同，该wait命令等待所有（不带任何参数）或特定后台任务完成。

6. time

   - time command1
   - 测量一个命令或一个脚本的运行时间

7. printf

   ```bash
   #!/usr/bin/bash
   read -p "please input username\'prefix & password & creat_num" prefix password num
   # 这里直接原样输出了多行内容
   printf "
   -------------
   user prefix：$prefix
   user password：$password
   user number: $num
   -------------
   "
   ```

   

8. 文件描述符（文件句柄）

   - 查看当前进程中，打开的文件描述符列表：`ll /proc/$$/fd`，fd——file description，$$——表示当前进程
   - 打开file1文件并且指定文件描述符序号：`exec 6<> /file1`
   - 关闭file1文件并且释放文件描述符：`exec 6>&-`
   - 其实我们修改文件就是在修改文件描述符
   - 当一个文件FD未被释放，删除源文件也不会影响FD，并且还可以通过fd恢复源文件

9. 匿名管道和命名管道

   - 管道对应的就是一个管道文件
   - 匿名管道，就像一般的管道命令，这个文件没有名字，暗送，eg：`rpm -qa | grep bash`
   - 命名管道，通过创建具体/path/file1文件来实现，eg：`mkfifo /tmp/tmpfifo`，
     - 可以实现不同终端之间的通信，一个端子向fifo文件里写数据，一个端子从fifo文件读数据。
   - 管道里的数据，一旦看了或用了就没了

10. watch

    - `watch -n command`
    - watch是周期性的执行下个程序，并全屏显示执行结果
    - eg：`watch -n2 ./count_tcpconn_status.sh`，间隔两秒钟，执行一次脚本

11. 



# log
