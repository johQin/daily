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

4. 安装相关软件包

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

   - 

5. 