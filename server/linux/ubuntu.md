# 1 快捷键

1. 切换输入法：【win（super） + space 】
2. win + R : 【Alt + F2】
3. 打开新的终端：【Ctrl + Alt + T】
4. 切换窗口：【Alt + Tab】
5. win + e : setting -> keyboard -> shortcuts , 添加，name随便写，command：nautilus，快捷键：super + e
6. firefox：截图 【ctrl + shift + s】

# 2 常用命令
1. 查看文件或文件夹大小：ls -hl

2. [apt-get详细](https://blog.csdn.net/qq_44885775/article/details/124278721)

3. [tar](https://www.runoob.com/w3cnote/linux-tar-gz.html)

   ```bash
   # 将所有 .jpg 的文件打成一个 tar 包，并且将其用 gzip 压缩，生成一个 gzip 压缩过的包，包名为 all.tar.gz。
   # tar 中使用 -z 这个参数来调用gzip。c——compress
   tar -czf all.tar.gz *.jpg
   # x——解压
   tar -xzf all.tar.gz
   ```

4. [linux查询文件名 或 文件内容中 包含特定字符串的所有文件](https://blog.csdn.net/weixin_40482816/article/details/121425903)

   - `find ./ -name '*2021-11-01*'`，查看当前文件夹（及子目录中）下，文件名包含2021-11-01的文件
   - `find ./ -name '*2021-11-01*' -maxdepth 1 `，查看当前文件夹下

5. [xargs](https://www.runoob.com/linux/linux-comm-xargs.html)：是给命令传递参数的一个过滤器，也是组合多个命令的一个工具。

   `find /sbin -perm +700 |xargs ls -l`，将前一个命令find的std ，通过xargs，输出给ls作参数。

6. 查找运行的进程中是否包含某个进程

   ```bash
   ps ajx | head -1 && ps ajx | grep 'ssd' 	# 带列名的展示
      PPID     PID    PGID     SID TTY        TPGID STAT   UID   TIME COMMAND
      2000   24119    2000    2000 ?             -1 Sl    1000   0:00 /usr/libexec/gvfsd-dnssd --spawner :1.2 /org/gtk/gvfs/exec_spaw/3
      6967  298419  298418    6967 pts/0     298418 S+    1000   0:00 grep --color=auto ssd
   
   ```

7. [service和systemctl的区别](https://blog.csdn.net/juanxiaseng0838/article/details/124123822)

   - service命令其实是去/etc/init.d目录下，去执行相关程序
   - systemctl是一个systemd工具，主要负责控制systemd系统和服务管理器。在/lib/systemd/system

8. 

# 3 ubuntu软件安装

1. [vmtool安装](https://blog.csdn.net/weixin_45035342/article/details/126638191)

   - vmtool安装后，可以实现宿主机与客户机【ctrl + c】和【Ctrl + v】互通。

   - 宿主机与客户机共享文件夹：在vmware中，菜单->虚拟机->设置->选项->共享文件夹，总是启用，添加文件夹（windows中的），可以在ubuntu系统的/mnt/hgfs中看到。

     

2. [画图工具](https://blog.csdn.net/xhtchina/article/details/122929567)

3. [安装docker](https://blog.csdn.net/u012563853/article/details/125295985)

4. [创建桌面快捷方式](https://blog.csdn.net/weixin_43031313/article/details/129385915)

5. [安装finalshell](https://blog.csdn.net/zhao001101/article/details/128002640)

6. [ubuntu安装git，并设置ssh](https://blog.csdn.net/qq_26849933/article/details/125062667)

7. [安装mysql](https://blog.csdn.net/weixin_39589455/article/details/126443521)

   - [安装mysql8.0](https://segmentfault.com/a/1190000039203507)

   - [navicat无限试用](https://www.xmmup.com/linuxubuntuxianavicat-premium-16dewuxianshiyong.html)

     naicat.AppImage文件需要用“磁盘映像挂载器”挂载到磁盘上，然后提取其中的png，以创建桌面快捷方式

     网络上有关于appImage文件通过 --appimage-extract进行解压，但在这里好像没有生效。

8. [ubuntu安装essayconnect](https://blog.csdn.net/weixin_37926734/article/details/123068318)

   - [essay安装后无法打开的问题](https://blog.csdn.net/u011426115/article/details/126660001)

9. [vi 编辑写入保存和退出](https://blog.csdn.net/qq_33093289/article/details/127915742)

   默认linux系统都有vi，而没有vim，安装vim：sudo apt install vim

   1. 插入
      - shfit+i 进入插入编辑文本模式
   2. 退出和保存
      - 按【ESC】键跳到命令模式

          　　1. 按【ESC】键跳到命令模式，然后再按【:】冒号键，最后再按【wq】，即可保存退出vi的编辑状态；
            　　2. 如果是不想保存直接按下【:】冒号键加【q!】键，就能直接退出，不保存；
              　　3. 此外还有这些命令，
               - :w 保存文件但不退出vi；
               - :w file 将修改另外保存到file中，不退出vi；
               - :w! 强制保存，不推出vi；
               - :wq 保存文件并退出vi；
               - :wq! 强制保存文件，并退出vi；
               - q: 不保存文件，退出vi；
               - :q! 不保存文件，强制退出vi；:e! 放弃所有修改，从上次保存文件开始再编辑；



# 4 c

1. [在ubuntu中配置c++开发环境](https://blog.csdn.net/qq_33867131/article/details/126540537)
   - [修改项目的环境](https://blog.csdn.net/qq_19734597/article/details/103056279)
   - file-->Settings-->Build,Execution,Deployment-->Toolchains，配置gcc，g++，make的位置

# 5 docker

1. [docker导入tar包作镜像](https://blog.csdn.net/blood_Z/article/details/126038450)

   - ```bash
     # 法一
     docker import  tar包名字.tar 镜像名称：版本id
     # 法二
     docker load -i tar包名字.tar
     ```

2. 





# 6 操作

1. linux设置开机自动执行脚本
   - 修改/etc/rc.d/rc.local（如果没有，则修改/etc/rc.local文件，再没有，就生成一个rc.local)，添加自定义的脚本至文件最后
   - 开启rc.local服务
   - rc.local服务使能。
   - 注意：脚本必须使用exit 0结束，
2. ubuntu系统的hosts（ip和域名映射）：/etc/hosts
3. [ubuntu 初次使用root身份登录](https://blog.csdn.net/weixin_56364629/article/details/124608110)
4. [ubuntu图像化界面不允许root用户登陆](https://blog.csdn.net/Ki_Ki_/article/details/128832659)
5. [ubuntu安装拼音输入法](https://blog.csdn.net/weixin_61275790/article/details/130787987)

​     
