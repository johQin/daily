# docker

docker是基于go语言实现的云开源项目。

docker的主要目标是“**Build, Ship and Run Any App, Anywhere** ”，也就是对应用组件的封装、分发、部署、运行等生命周期的管理，使用户App（可以是Web应用或数据库应用等等）及其运行环境能够做到“**一次镜像，处处运行**”。

linux的容器技术的出现，就解决了这样一个问题，而docker就是在它的基础上发展过来的。将应用打包成镜像，通过镜像成为运行在docker容器上的实例，而docker容器在任何操作系统都是一致的，这就实现了跨平台，跨服务器。只需要一次配置好环境，换到别的主机上，就可以一键部署，大大简化了操作。

docker是解决了运行环境和配置问题的软件容器，方便做集成并有助于整体发布的容器虚拟化技术。

docker是一个能够把开发的应用程序自动部署到容器的开源引擎。

docker设计的目的就是要加强开发人员写代码的开发环境与应用程序要部署的生产环境的一致性，从而降低"开发时一切正常，发布就出问题"的风险。

# 1 介绍

## 1.1 [docker与虚拟机的区别](https://zhuanlan.zhihu.com/p/351621747)



### 1.1.1 架构区别

![](./figure/虚拟机与docker的区别.jpg)

docker：物理机-操作系统-docker-APP（container）

虚拟机：物理机-管理程序 hypervisor（vmware、kvm...）-vm操作系统-APP

两者对比来看docker比虚拟机少了一层vm的操作系统。docker的APP是直接运行在宿主机上的，而虚拟机的APP是运行在虚拟在宿主机上的操作系统上的。

#### 容器技术

**linux容器是，与系统其他部分隔离开的一系列进程。从另一个镜像运行，并由该镜像提供支持进程所需的全部文件，容器提供的镜像包含了应用的所有依赖项，因而在从开发到测试再到生产的整个过程中，它都具有可移植性和一致性。**

linux容器不是模拟一个完整的操作系统，而是对进程进行隔离，有了容器，就可以将软件运行所需的所有资源打包到一个隔离的容器中。容器与虚拟机不同，不需要捆绑一套操作系统，只需要软件工作所需的库资源和设置。系统因此变得高效轻量并保证部署在任何环境中的软件都能始终如一的运行。

**docker容器是在操作系统层面上实现虚拟化，直接复用本地主机的操作系统，而传统的虚拟机则是在硬件层面实现虚拟化**

容器内的应用程序直接运行于宿主的内核，容器内没有自己的内核，且没有进行硬件虚拟，因此容器要比传统虚拟机更为轻便。

每个容器之间相互隔离，每个容器有自己的文件系统，容器之间进行不会相互影响，能区分计算资源。

### 1.1.2 其他区别

| 比较项               | docker           | vm                                |
| -------------------- | ---------------- | --------------------------------- |
| 启动时间             | 秒级（启动应用） | 分钟级（启动操作系统 + 启动应用） |
| 存储占用             | MB（应用的大小） | GB（操作系统 + 应用的大小）       |
| 性能                 | 接近原生         | 弱于原生                          |
| 单个宿主机支持的数量 | 上千个           | 几十个                            |



## 1.2 docker组件

### 1.2.1 镜像image

docker镜像就是一个只读模板，镜像可以用来创建Docker容器，一个镜像可以创建很多容器。

镜像有时候就像容器的"源代码"。

类似于java的class，而docker容器就类似于java中，new出来的实例对象。

也相当于一个最小文件系统。

| docker | java面向对象 |
| ------ | ------------ |
| 容器   | 对象实例     |
| 镜像   | 类           |

### 1.2.2 容器container

#### 从面相对象的角度理解

docker利用容器独立运行一个或一组应用，应用程序或服务运行在容器里面。

容器就相当于一个虚拟化的运行环境，容器是用镜像创建的运行实例。

镜像是docker生命周期中构建或打包阶段，而容器则是启动或执行阶段。

镜像是静态的定义，容器是镜像运行时的实体。

容器为镜像提供了一个标准的和隔离的运行环境，它可以被启动，开始，停止，删除。

#### 从镜像容器角度

可以把容器看做是一个简易版的linux环境（最小，最核心的linux内核文件，不需要的不加载，包括root用户权限，进程空间，用户空间和网络空间等），和运行在其中的应用程序。

### 1.2.3 仓库repository

仓库是集中存放镜像文件的地方。

类似于：

maven仓库，存放各种jar包

github仓库，存放各种git项目的地方

docker公司提供的官方registry被称作docker hub，存放各种镜像模板的地方。



仓库分为公开仓库（Public）和私有仓库（Private）两种形式。

最大的公开仓库是docker Hub（https://hub.docker.com/)。存放了数量庞大的镜像供用户下载。

国内的公开仓库包括阿里云、网易云等。

### 1.2.4 引擎

docker本身是一个容器运行载体或称之为管理引擎（docker daemon）。

我们把应用程序和配置依赖打包好形成一个可交付的运行环境，这个打包好的运行环境就是image镜像文件。只有通过这个镜像文件才能生成Docker容器实例。

![](./figure/docker入门架构.PNG)

docker是一个client-server结构的系统，Docker守护进程运行在主机上，然后通过socket连接从客户端访问，守护进程从客户端接受命令并管理运行在主机上的容器。

容器，是一个运行时环境，是一个个的集装箱。

## 1.3 docker架构

docker是一个C/S模式的架构，后端是一个松耦合架构，多个模块各司其职。

docker运行的基本流程：

1. 用户是使用docker Client与docker Daemon建立通信，并发送请求给后者。
2. docker Daemon作为Docker架构中的主体部分，首先提供Docker server的功能使其可以接受docker client的请求。
3. docker Engine执行docker内部的一系列工作，每一项工作都是一系列job的形式的存在
4. job运行的过程中，当需要容器镜像时，则从docker registry中下载镜像，并通过镜像管理驱动Graph的形式存储。
5. 当需要docker创建网络环境时，通过网络管理驱动network driver创建并配置docker容器网络环境
6. 当需要限制docker容器运行资源或执行用户指令操作时，则通过exec driver来完成。
7. Libcontainer则是一项独立的容器管理包，network driver以及exec driver都是通过Libcontainer来实现具体对容器进行操作

![](./figure/docker架构.jpg)

## 1.4 docker安装

https://docs.docker.com/engine/install/centos/

步骤：

1. 确定centos7及以上版本

   - ```bash
     cat  /etc/redhat-release
     CentOS Linux release 8.0.1905 (Core) 
     ```

2. 卸载旧版本

   - ```bash
     sudo yum remove docker \
                       docker-client \
                       docker-client-latest \
                       docker-common \
                       docker-latest \
                       docker-latest-logrotate \
                       docker-logrotate \
                       docker-engine
     ```

3. yum安装gcc

   - gcc 可以编译c/c++

   - ```bash
     # 查看是否通过yum安装过gcc
     yum list installed | grep gcc
     # 如果安装过，那么将会出现下面的两行
     gcc.x86_64                            8.5.0-4.el8_5                     @AppStream
     gcc-c++.x86_64                        8.5.0-4.el8_5                     @AppStream
     
     # 如果没有安装，通过下面的两个命令安装
     yum -y install gcc
     yum -y install gcc-c++
     ```

4. 设置stable镜像仓库

   - ```bash
     # yum-utils 提供了yum-config-manager工具
     yum install -y yum-utils
     
     # 通过yum-config-manager安装stable repository，这个库地址需要改为国内库地址，
     # 如果连的是国外的库，
     # 比如docker官网 https://download.docker.com/linux/centos/docker-ce.repo
     # 经常会出现网络超时
     # 报以下的错误：
     # [Error 14] curl#35 - TCP connection reset by peer
     # [Error 12] curl#35 - Timeout
     
     # 推荐使用阿里云，腾讯云，华为云，网易云的
     # 
     yum-config-manager \
         --add-repo \
         http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
         
     ```

5. 更新yum软件包索引

   ```bash
   # centos 8 需要去掉fast执行
   yum makecache fast
   ```

6. 安装docker ce

   ```bash
   yum -y install docker-ce docker-ce-cli containerd.io
   ```

7. 启动docker

   ```bash
   systemctl start docker
   ```

8.  测试

   ```bash
   docker version
   
   # Client: Docker Engine - Community
   # Version:           20.10.17
   # API version:       1.41
   # Go version:        go1.17.11
   # Git commit:        100c701
   # Built:             Mon Jun  6 23:03:11 2022
   # OS/Arch:           linux/amd64
   # Context:           default
   # Experimental:      true
   
   # Server: Docker Engine - Community
   # Engine:
   #  Version:          20.10.17
   #  API version:      1.41 (minimum version 1.12)
   #  Go version:       go1.17.11
   #  Git commit:       a89b842
   #  Built:            Mon Jun  6 23:01:29 2022
   #  OS/Arch:          linux/amd64
   #  Experimental:     false
   # containerd:
   #  Version:          1.6.6
   #  GitCommit:        10c12954828e7c7c9b6e0ea9b0c02b01407d3ae1
   # runc:
   #  Version:          1.1.2
   #  GitCommit:        v1.1.2-0-ga916309
   # docker-init:
   #  Version:          0.19.0
   #  GitCommit:        de40ad0
   
   
   docker run hello-world
   
   
   docker run hello-world
   Unable to find image 'hello-world:latest' locally
   latest: Pulling from library/hello-world
   2db29710123e: Pull complete 
   Digest: sha256:53f1bbee2f52c39e41682ee1d388285290c5c8a76cc92b42687eecf38e0af3f0
   Status: Downloaded newer image for hello-world:latest
   
   Hello from Docker!
   This message shows that your installation appears to be working correctly.
   
   To generate this message, Docker took the following steps:
    1. The Docker client contacted the Docker daemon.
    2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
       (amd64)
    3. The Docker daemon created a new container from that image which runs the
       executable that produces the output you are currently reading.
    4. The Docker daemon streamed that output to the Docker client, which sent it
       to your terminal.
   
   To try something more ambitious, you can run an Ubuntu container with:
    $ docker run -it ubuntu bash
   
   Share images, automate workflows, and more with a free Docker ID:
    https://hub.docker.com/
   
   ```

9.  卸载docker

   ```bash
   systemctl stop docker
   yum remove docker-ce docker-ce-cli containerd.io
   rm -rf /var/lib/docker
   rm -rf /var/lib/containerd
   ```

   

10. 阿里云镜像加速器

    ![](./figure/阿里云镜像加速器.png)

    ```bash
    # 后面为了拉镜像，启动容器更快一点，必须配置一个阿里云的镜像加速器
    
    # 注册一个阿里云账户，阿里云为了把云生态做起来，给个体开发者了一些便利
    
    # 阿里官网的操作
    mkdir -p /etc/docker
    # tee命令用于将标准输入复制到指定文件，并显示到标准输出。tee指令会从标准输入设备读取数据，将其内容输出到标准输出设备，同时保存成文件。
    # 镜像的地址记得要写成自己的，你如果直接从你的那个地方粘贴，那么你的地址就是你的加速器地址
    tee /etc/docker/daemon.json <<-'EOF'
    {
      "registry-mirrors": ["https://ocx43prw.mirror.aliyuncs.com"]
    }
    EOF
    
    systemctl daemon-reload
    systemctl restart docker
    ```

    

11. 

# 2 docker 常用命令

## 2.1 帮助启动类命令

```bash
# 1.启动docker
systemctl start docker
# 2.停止docker
systemctl stop docker
# 3.重启docker
systemctl restart docker
# 4.查看docker状态
systemctl status docker
# 5.开启启动
systemctl enable docker
# 6.查看docker概要信息
docker info
# 7.查看docker帮助文档
docker help
# 8.查看docker命令帮助文档
docker <command> --help

```

## 2.2 镜像命令

```bash
# 1.列出本地主机上的镜像
docker images -[aq]
REPOSITORY    TAG       IMAGE ID       CREATED        SIZE
hello-world   latest    feb5d9fea6a5   9 months ago   13.3kB
# repository：表示镜像仓库源
# tag：镜像的标签版本号，同一仓库源可以有多个tag版本，我们可以使用repository:tag来定义不同的镜像
# 如果你不指定某个镜像的版本标签，那么docker默认使用latest镜像
# image id： 镜像id
# created： 镜像创建时间
# size 镜像的大小
# 参数：
-a 列出本地所有镜像(含历史映像层)
-q 只列出image id

# 2.镜像搜索
docker search image_name
docker search nginx
# 查出前5个镜像，默认limit是25个
docker search --limit 5 nginx
NAME        DESCRIPTION                         STARS     OFFICIAL   AUTOMATED
nginx      Official build of Nginx.             17124     [OK]       
linuxserver/nginx  An Nginx container, brought to you by LinuxS…   169                  
bitnami/nginx   Bitnami nginx Docker Image      137      				 [OK]
ubuntu/nginx     Nginx, a high-performance reverse proxy & we…   55                   
bitnami/nginx-ingress-controller   Bitnami Docker Image for NGINX…  19			 [OK]

# 3.下载镜像
docker pull image_name # 等同于docker pull image_name:latest
docker pull image_name:tag # 下载指定版本的镜像

# 4.查看镜像/容器/数据卷所占空间
docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          1         1         13.26kB   0B (0%)
Containers      2         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B

# 5.删除镜像
docker rmi image_name_or_image_id[:tag] # rm-remove i-image，
docker rmi -f image_name_or_image_id[:tag] # 当本地正在运行镜像的容器时，执行上面的命令可能无法删除，这时需要用到-f，强制删除
docker rmi img1:tag1 img2:tag2 # 删除多个镜像
docker rmi -f $(docker images -qa)

#docker commit / docker push
```

虚悬镜像：仓库名和版本名都是\<none>镜像

## 2.3 容器命令

下面打算安装一个ubuntu，意思就是在centos操作系统上的docker里面安装一个Ubuntu。

### 2.3.1 启动容器

```bash
# 1.下载镜像
docker pull ubuntu
# 2.新建并启动容器
docker run [option] image [command]
# option :

# --name="container_name" 为容器指定名称
# -d 后台运行容器，并返回容器id，即：启动守护式容器

# -i interactive，以交互模式运行容器，通常与-t同时使用
# -t 为容器重新分配一个伪输入终端（tty），通常与-i同时使用
# 启动一个交互式容器

# -P 随机端口映射

# -p 指定端口映射
# -p hostPort:containerPort，如果访问宿主机的hostPort端口，就会映射到docker内部的containerPort端口，container端口也就是内部应用（鲸鱼背里的集装箱）监听的端口，例如nginx就监听的是80端口
# -p ip:hostPort:containerPort

docker run -it ubuntu /bin/bash
# 使用镜像ubuntu:latest，并以交互模式启动一个容器，在容器内执行/bin/bash命令
# -it 启动交互式终端
# ubuntu 镜像名称
# /bin/bash 通过这个命令，可以运行linux shell脚本命令，这个终端里面只能运行linux内核的一些命令，因为安装的镜像只有70多M，是一个内核的精简的ubuntu系统

# 命令执行成功以后，出现的端口就是ubuntu的终端了
# 输入exit 退出container，同时容器也关闭了

[root:11:49@~]docker run -it --name=ubuntu1 ubuntu
root@1ee74ac3f7e9:/ ls
bin  boot  dev  etc  home  lib  lib32  lib64  libx32  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var


# 3.查看所有容器启动运行情况
docker ps
# -a 列出当前所有的容器 ，包括启动 + 未启动的
# -l 列出最近创建的容器
# -n 显示最近n个创建的容器
# -q 静默模式，只显示容器编号
CONTAINER ID   IMAGE     COMMAND   CREATED            STATUS              PORTS     NAMES
1ee74ac3f7e9   ubuntu    "bash"    About a minute ago   Up About a minute         ubuntu1

# 4.退出容器
# 两种退出方式
# 方式一：exit，run进去容器，exit退出，容器停止
# 方式二：ctrl + p + q， run进去容器，ctrl+p+q退出，容器不停止

# 5.启动已停止的容器
docker start container_id_or_container_name
# 6.重启容器
docker restart container_id_or_container_name
# 7.停止容器
docker stop container_id_or_container_name
# 8.强制停止容器
docker kill container_id_or_container_name
# 9.删除容器
docker rm container_id_or_container_name # 删除已停止的容器
docker rm -f container_id_or_container_name # 强制删除容器，容器可以正在运行

```

### 2.3.2  启动守护式容器

在2.3.1中我们可以知道，退出容器有两种方式，通过ctrl+p+ q即可退出，但容器并不停止。

```bash
# 1. 以后台模式运行一个容器
docker run -d ubuntu
# 查看运行情况
docker ps -a
# 发现当前容器已经退出了

# docker 容器后台运行必须要一个前台进程
# 容器运行的如果不是那些一直挂起的命令（例如top，tail），就是会自动退出的，这是docker机制决定的。
# 所以如果要启动守护式容器，那么就必须run 一些具备前台进程的镜像，例如redis，

# 所以最佳的解决方式是：将你要运行的程序以前台进程的形式运行。
# 常见的就是交互命令行模式（-it），然后ctrl + p + q退出

# 2.查看容器运行日志
docker log container_id

# 3. 查看容器内运行的进程
docker top container_id
# docker里跑起来的容器都是一个个linux系统
# 可以把容器看做一个简易版的linux环境
UID    PID           PPID         C            STIME        TTY        TIME          CMD
root   465017        464998       0            11:50        pts/0      00:00:00     bash


# 4.查看容器内部细节
docker inspect container_id
# 可以查看网络，ip配置等等容器细节（简易版linux的容器细节）
```

### 2.3.3 进入正在运行的容器

重新进入正在运行的容器，并以命令行交互。

```bash
# 5. 进入正在运行的容器
# 两种方式
# 方式1
docker exec -it container_id /bin/bash
# 方式2
docker attach container_id
# 区别：
# exec：是在容器中打开新的终端，并且可以启动新的进程，用exit退出，不会导致容器的停止
# attach：直接进入容器启动命令终端，不会启动新的进程，用exit退出，会导致容器的停止

# 对容器里的文件进行备份
#6. copy容器里的文件到宿主机
docker cp container_id:container_inner_path host_path

# 对整个容器进行备份，包括容器里的文件
#7. 将容器导出为一个tar文件
docker export container_id > container_file.tar
#8. import从tar包中的内容创建一个新的文件系统再导入为镜像
cat container_file.tar | docker import - 用户名/镜像名:版本号
# 用户名镜像名版本号随意命名
docker images 
# 可以查看导入后的镜像已经在列表中了
```

![](./figure/docker_command.png)



# 3 docker 镜像

镜像是一种轻量级、可执行的独立软件包，它包含运行某个软件所需的所有内容，我们把应用程序和配置依赖打包好形成一个可交付的运行环境(包括代码、运行时需要的库、环境变量和配置文件等)，这个打包好的运行环境就是image镜像文件。

## 3.1 镜像分层

镜像是一个分层的文件系统。

### 3.1.1 联合文件系统

**docker的镜像实际上由一层一层的文件系统组成，这一层层的文件系统共同构成一个联合文件系统。**

Union文件系统（UnionFS）是一种**分层、轻量级并且高性能**的文件系统，它支持对文件系统的修改作为一次提交来**一层层**的叠加，同时可以将**不同目录挂载到同一个**虚拟文件系统下。

Union 文件系统是 Docker 镜像的基础。镜像可以通过分层来进行继承，基于基础镜像（没有父镜像），可以制作各种具体的应用镜像。

特性：一次同时加载多个文件系统，但从外面看起来，只能看到一个文件系统，联合加载会把各层文件系统叠加起来，这样最终的文件系统会包含所有底层的文件和目录

### 3.1.2 镜像层次与linux

镜像底层是一个引导文件系统bootfs， bootfs主要包含boot loader 和 linux kernel，bootloader主要是用于引导加载kernel。

inux系统刚启动时也会加载bootfs，所以在这一层上，镜像和我们的linux/unix系统是一样的。

当boot加载完kernel以后，整个内核就在内存中了，此时内存的使用权就会移交给内核，此时系统就会卸载bootfs。

rootfs在bootfs之上，包含的就是典型linux系统中的/dev、/proc、/bin、/etc等标准目录和文件。rootfs就是各种不同操作系统的发行版，比如Ubuntu，centos等。

![](./figure/image分层.jpg)

有时候，镜像底层可以直接使用host的kernel，所以镜像自己直接提供rootfs就行。

镜像分层最大的好处就是共享资源，方便复制迁移，就是为了复用。

比如说，多个镜像都是从相同的base镜像而来，那么docker host 只需要在磁盘上保存一份base镜像即可。同时内存中也只需加载一份base镜像，就可以为所有容器服务了，而且镜像的每一层都可以被共享。

如果我想做一个包含vim的ubuntu镜像，所以只需要从基础ubuntu镜像扩展vim功能即可，没必要从头再来。

### 3.1.3 容器层与镜像层

docker的镜像层都是只读的，容器层是可写的。当容器启动时，一个新的可写层被加载到镜像的顶部，这一层通常被称作容器层，容器层之下都是镜像层。

![](./figure/容器层与镜像层.jpg)

## 3.2 制作镜像commit

提交容器副本使之成为一个新的镜像

例程：对官方的ubuntu扩展vim功能

```bash
# 进入ubuntu容器内部，并打开交互式的命令行（-it）
# 1.更新ubuntu的包管理工具，类似于centos的yum命令
apt-get update
# 2.通过包管理工具安装vim
apt-get -y install vim
# 退出容器ctrl + p + q
# 3.提交容器副本使之成为一个新的镜像
docker commit -m="ubuntu extend vim" -a='khq' 702c56fdc2fa ubuntuVim:0.1

docker images
REPOSITORY             TAG       IMAGE ID       CREATED          SIZE
sifang/ubuntuwithvim   1.1       4e56eda4bb5e   11 seconds ago   178MB
tomcat                 latest    fb5657adc892   7 months ago     680MB
ubuntu                 latest    ba6acccedd29   9 months ago     72.8MB
hello-world            latest    feb5d9fea6a5   10 months ago    13.3kB

# 可以看到新的镜像它的容量变大为178m

```

## 3.3 发布镜像

![](./figure/发布镜像.jpg)

镜像生成方法：dockerfile 和 基于容器commit镜像

### 3.3.1 发布到阿里云

选择控制台=>容器镜像服务=>实例列表=>个人实例=>命名空间=>创建命名空间，创建仓库

![](./figure/发布镜像.png)

这里的仓库名就是你的镜像名，而不是可以存放多个镜像的仓库名，所以下面的daily应实际为ubuntuVim

然后在仓库页面中就会有相应的操作引导，让你的docker服务器与aliyun建立连接

```bash
docker login --username='' registry.cn-hangzhou.aliyuncs.com
password:*****
# 密码可以在个人实例=>访问凭证那里获取
Login Succeeded
# 标记镜像版本号
docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/sifang/daily:[镜像版本号]
# 推送镜像到阿里云
docker push registry.cn-hangzhou.aliyuncs.com/sifang/daily:[镜像版本号]
The push refers to repository [registry.cn-hangzhou.aliyuncs.com/sifang/daily]
41e8d863aa21: Pushed 
9f54eef41275: Pushed 
1.1: digest: sha256:d1f39fbda865ee5a98ecdd2b37224f3d58081a0a1e347be4eda581af743306a5 size: 741

# 删除本地镜像
docker rmi -f 4e56eda4bb5e

Untagged: sifang/ubuntuwithvim:1.1
Untagged: registry.cn-hangzhou.aliyuncs.com/sifang/daily:1.1
Untagged: registry.cn-hangzhou.aliyuncs.com/sifang/daily@sha256:d1f39fbda865ee5a98ecdd2b37224f3d58081a0a1e347be4eda581af743306a5
Deleted: sha256:4e56eda4bb5efc1fab1591a83cecc3c36d97159a020894056ecee25448757fd4

# 拉取远程服务器镜像
docker pull registry.cn-hangzhou.aliyuncs.com/sifang/daily:1.1
1.1: Pulling from sifang/daily
7b1a6ab2e44d: Already exists 
5579d8edfa40: Already exists 
Digest: sha256:d1f39fbda865ee5a98ecdd2b37224f3d58081a0a1e347be4eda581af743306a5
Status: Downloaded newer image for registry.cn-hangzhou.aliyuncs.com/sifang/daily:1.1
registry.cn-hangzhou.aliyuncs.com/sifang/daily:1.1
[root:17:47@~]docker images
REPOSITORY          TAG       IMAGE ID       CREATED             SIZE
.../sifang/daily   1.1       4e56eda4bb5e   About an hour ago   178MB
tomcat         latest    fb5657adc892   7 months ago        680MB
ubuntu         latest    ba6acccedd29   9 months ago        72.8MB
hello-world    latest    feb5d9fea6a5   10 months ago       13.3kB

# 再次运行镜像
docker run -it 4e56eda4bb5e /bin/bash

```

### 3.3.2 发布到自己的docker私有库