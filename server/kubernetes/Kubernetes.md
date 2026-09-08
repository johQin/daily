

# Kubernetes

Kubernetes （简写：K8S），在希腊语意为“舵手”或“驾驶员”

k8s是谷歌在2014年开源的容器化集群管理系统

目标：让部署容器化应用更加简洁和高效

# 0 绪论





## 0.1 功能

1. 基于容器对应用运行环境的资源配置要求，自动部署应用容器
2. 自我修复：
   - 容器启动失败，会进行重启
   - 部署的节点有问题时，会对容器进行重新部署和重新调度
   - 当容器未通过监控时，会关闭此容器直到容器正常运行时，才会对外提供服务。
3. 水平扩展：通过简单的命令、用户UI界面或基于CPU等资源使用情况，对应用容器进行规模扩大或裁剪
4. 服务发现：内置服务发现和负载均衡
5. 滚动更新：根据应用的变化，对应用容器运行的应用，进行一次性或批量式更新
6. 版本回退
7. 密钥和配置管理
8. 存储编排：自动实现存储系统挂载及应用
9. 批处理：一次性任务，定时任务



## 0.2 [集群架构](https://developer.aliyun.com/article/1635071)

![](./legend/k8s集群架构图.png)

### Master Node

- 也称Control  Plane

- k8s 集群控制节点，对集群进行调度管理，接受集群外用户去集群操作请求

- Master Node 由API Server、Scheduler、ClusterState Store（ETCD 数据库）和 Controller MangerServer 所组成

- **API Server**

  - 集群的统一入口，暴露 Kubernetes API

  - 功能：
    - **验证和授权**：负责验证用户的身份，并根据配置的访问控制策略进行授权，确保只有合法的请求才能对集群进行操作。
    - **集群状态管理**：集群状态信息都通过它来访问和更新
    - **通信枢纽**：集群的通信桥梁
  - 当用户或其他组件向 Kube-API Server 发送请求时，API Server 首先进行身份验证和授权检查，然后对请求的数据进行验证和处理。处理完成后，API Server 会将数据存储到 etcd 中，同时通知其他组件进行相应的操作。

- **Etcd**

  - Etcd 是一个分布式键值存储，用于存储 Kubernetes 集群的所有数据，所有的配置信息、状态信息都存储在 etcd 中。
  - 功能：
    - **数据存储**：Etcd 负责存储所有的集群状态数据，包括 Pod、Service、ConfigMap、Secret 等。
    - **数据可靠性**：Etcd 通过分布式架构保证数据的高可用性和一致性，确保集群状态数据在发生故障时仍能可靠存储。
    - **数据访问**：Etcd 提供了高效的键值对存储和访问接口，支持高频率的读写操作
  - API Server 通过 etcd API 进行数据的读写操作。当集群状态发生变化时，APIServer 会将新的状态数据写入 etcd，同时其他组件可以监听 etcd 的变化，从而进行相应的处理。

- **Scheduler**

  - Scheduler 是 k8s的调度组件，负责将新创建的 Pod 调度到合适的工作节点上。它根据预设的调度策略和节点状态，选择最合适的节点来运行 Pod。
  - 功能：
    - 资源调度：Scheduler 根据节点的资源使用情况和 Pod 的资源需求，选择合适的节点来运行 Pod。
    - 策略配置：支持多种调度策略，包括资源优先、亲和性、反亲和性等，用户可以根据需求自定义调度策略。
    - 负载均衡
  - Scheduler 通过监听 Kube-API Server 上的调度请求，获取需要调度的 Pod 列表。然后，它根据预设的调度策略和节点的状态，选择最合适的节点，并将调度结果写回 API Server，最终由相应的节点来运行 Pod。

- **Controller-Manager**

  - Controller-Manager 是 Kubernetes 控制平面的控制管理组件，负责管理集群的各种控制器。这些控制器是用于处理集群状态变化的后台进程。
  - 功能：
    - **控制器管理**：包括 Node Controller、Replication Controller、Endpoint Controller、Namespace Controller 等，这些控制器分别负责节点管理、副本管理、服务发现、命名空间管理等功能。
    - **自动化操作**：控制器通过监听集群状态变化，自动执行相应的操作，如副本调整、故障节点隔离、服务更新等。
    - **一致性保证**：通过控制器的自动化操作，Kube-Controller-Manager 保证了集群状态的一致性和可靠性。
  - Controller-Manager 通过监听 API Server 的事件，获取集群状态变化的信息。根据不同的控制器，它会执行相应的操作，如创建或删除 Pod、副本调整、节点故障处理等，并将结果写回 API Server，从而更新集群状态。

### Worker Node

- **Kubelet**
  - 是运行在每个工作节点上的主要代理进程，负责管理节点上的 Pod 和容器。它通过与 Kube-APIServer 交互，确保节点上容器的正确运行。
  - 功能：
    - **Pod 管理**：Kubelet 负责启动和停止节点上的 Pod，并监控它们的状态，确保每个 Pod 按照预期运行。
    - **状态报告**：Kubelet 定期向 Kube-API Server 报告节点和 Pod 的状态，包括资源使用情况、健康状态等。
    - **配置管理**：Kubelet 根据从 Kube-API Server 获取的配置信息，配置和管理节点上的容器运行环境。
  - Kubelet 通过监听 Kube-APIServer 的调度信息，获取需要在本节点上运行的 Pod 列表。它根据 Pod 的配置文件，调用容器运行时（如 Docker、containerd）来启动和管理容器。同时，Kubelet 会定期向 Kube-APIServer 发送心跳信号和状态报告，确保控制平面能够及时了解节点和 Pod 的运行状况。
- **kube-proxy**
  - Kube-Proxy 是 Kubernetes 中的网络代理服务，运行在每个工作节点上，负责维护网络规则，管理 Pod 间的网络通信和负载均衡。
  - 功能：
    - **服务发现**：Kube-Proxy 负责维护节点上的网络规则，确保服务 IP 和端口能够正确映射到相应的 Pod 上。
    - **负载均衡**：Kube-Proxy 通过 IP Tables 或 IPVS 实现服务的负载均衡，将请求分发到后端的多个 Pod上。
    - **网络路由**：Kube-Proxy 处理网络流量，确保节点内外的通信能够正确路由到目标 Pod。
  - Kube-Proxy 通过监听 Kube-API Server 获取服务和端点的变化信息，然后根据这些信息动态更新节点上的网络规则。它使用 IP Tables 或 IPVS 来实现网络流量的转发和负载均衡，确保请求能够正确分发到相应的 Pod 上。
- **Container Runtime**
  - 容器运行时（Container Runtime）是 Kubernetes 中用于运行和管理容器的组件，常见的容器运行时有 Docker、containerd、CRI-O 等。
  - 功能：
    - **容器管理**：容器运行时负责启动、停止和监控容器的运行状态。
    - **资源隔离**：容器运行时通过 cgroup、namespace 等机制实现容器的资源隔离和限制。
    - **镜像管理**：容器运行时负责从镜像仓库拉取容器镜像，并在节点上进行存储和管理。
  - Kubelet 通过 CRI（Container Runtime Interface）与容器运行时进行交互，向其发送启动和停止容器的指令。容器运行时根据这些指令，调用底层操作系统的容器技术（如 cgroup、namespace）来管理容器的生命周期和资源使用。同时，容器运行时还负责从镜像仓库拉取和管理容器镜像，确保容器能够按需启动。

## 0.3 核心概念



### 0.3.1 Pod

**Pod 是 K8s 最小、最基础的调度单元（运行单元）**，也是 K8s 调度、部署、管理的**最小原子**。

一个Pod里面可以包含1个或多个容器。

同一个Pod内的容器有以下特性：

1. 共享特性
   - 共享网络命名空间
     - 整个Pod共用同一个IP、同一个网卡、同一个端口空间
     - 容器之间直接用`localhost:port`就能相互访问
     - 不能在同一个Pod里占用相同端口，端口会冲突
   - 共享PID命名空间：默认不开启，配置`shareProcessNamespace: true`，容器能相互看到对方进程，可以互相查看，调试进程
   - 共享存储卷：Pod挂载的卷，所有容器都能挂载使用，实现文件共享，日志共享
   - 共享主机名，域名
   - 统一生命周期：同时创建，同时销毁，同时调度
     - 不会单独重启Pod里某一个容器，任一容器异常，可触发Pod重启
     - 调度时整体被调度到某一个节点，不会拆分到不同机器
2. 约束特性
   - 资源约束：CPU / 内存 是 Pod 维度整体限制，内部多个容器瓜分 Pod 分配的资源。
   - 同一节点绑定：同一个 Pod永远只会跑在同一个 K8s 节点，不会跨节点拆分。
   - 日志与隔离：每个容器的日志独立，归属同一个Pod。**网络和存储互通，但文件系统隔离，只能通过Volume共享文件**

### 0.3.2 Controller

**Controller 是管理单元**，负责：创建、扩缩容、重启、自愈、版本更新 Pod

**从不直接创建日常业务 Pod**，都是创建 Controller，由 Controller 帮你生成并维护 Pod。

| Controller类型 | 管理 Pod 特点                          | 适用场景                       |
| -------------- | -------------------------------------- | ------------------------------ |
| Deployment     | 无状态、随机 Pod、可随意重建           | web 服务、后端接口（90% 业务） |
| StatefulSet    | 有状态，有固定名称、固定网络标识、有序 | MySQL、Redis、MQ 有状态中间件  |
| DaemonSet      | 每个节点自动跑一个 Pod                 | 日志收集、监控代理、节点 agent |
| Job            | 跑完就退出的 Pod                       | 批量任务、数据备份             |
| CronJob        | 定时生成 Job 再创建 Pod                | 定时脚本、定时报表             |

### 0.3.3 Service

 Service 是 将运行在一个或一组 Pod上的网络应用程序公开为网络服务的方法。

Pod有两个致命的问题

- Pod随时会被销毁，重建，重建后IP会变
- 一个 Controller 管理多个 Pod 副本，**需要统一入口访问**

Service作用：

- 在集群内，给一组Pod提供固定不变的虚拟IP（Cluster IP）

- 在集群内，用Service名称DNS直接访问，不用记IP（K8s 集群里自带 **CoreDNS**，相当于集群内部的「专属 DNS 服务器」。

  所有 Pod、Service 都会被 CoreDNS 自动解析成域名）

- 做负载均衡，把请求转发给正常的pod

Service 通过 **selector 标签** 关联 Pod：

- Deployment Controller 给 Pod 打标签 `app=web`
- Service 配置 selector `app=web`
- Service 自动匹配所有带这个标签的 Pod，纳入后端转发池

| Service类型  | 特点                                                         |
| ------------ | ------------------------------------------------------------ |
| ClusterIP    | 集群内部虚拟 IP，**只能集群内互相访问**，最常用。            |
| NodePort     | **给集群里每一台宿主机，都开放同一个固定端口**，外部可以通过任意一台 **宿主机 IP:NodePort 端口** 访问到后端 Pod。<br />结构链路：外部用户 → 任意节点宿主机 IP:NodePort 端口 → Service → 负载均衡 → 后端 Pod |
| LoadBalancer | 对接云厂商负载均衡，分配公网 IP，对外暴露服务                |
| ExternalName | 把 **K8s 内部 Service 域名** 映射到**外部外网域名 / 内部自建域名**。 |

### 0.3.4 三者之间的关系

**层级关系：客户端 → Service → Controller → Pod**

三者之间的联系：

1. **Controller 管 Pod**：负责创建、启停、扩容、自愈 Pod；
2. **Service 罩住一组 Pod**：通过标签选中 Pod，对外提供固定访问入口 + 负载均衡；
3. **访问从不直接连 Pod**：永远连 Service，底层 Pod 随便重建、漂移、扩容，业务无感知。



## 0.4 硬件要求

测试环境：

- master：2核/4G/20G
- worker：4核/8G/40G





# 1 搭建k8s集群

两种方式：

- kubeadm（kubernetes-admin）
  - master 节点`kubeadm init`，初始化集群
  - worker节点`kubeadm join`，加入集群
  - kubeadm降低了部署门槛，但屏蔽了很多细节，遇到问题很难排查
- 组件部署方式：
  - 从github下载发行版的组件包，手动部署每个组件，组成k8s集群
  - 手动部署较为麻烦，但可以学习很多工作原理，也利于后期维护



## 1.1 二进制套件方式

### 1.1.1 linux 系统环境准备

[部署参考](https://gitee.com/moxi159753/LearningNotes/tree/master/K8S/3_%E4%BD%BF%E7%94%A8kubeadm%E6%96%B9%E5%BC%8F%E6%90%AD%E5%BB%BAK8S%E9%9B%86%E7%BE%A4)

```bash
# 关防火墙
# 查看防火墙状态
sudo ufw status
# centos
# 临时关 systemctl stop firewalld
# 永久关 systemctl disable firewalld
# ubuntu
# 临时关
sudo systemctl stop ufw
# 永久关
sudo ufw disable
# 关闭防火墙的原因：
# Kubernetes 的网络代理 kube-proxy 依赖 iptables 或 IPVS 来管理网络规则。而像 firewalld 这样的防火墙服务，可能会在背后操作 nftables，与 kube-proxy 管理的 iptables 规则产生冲突，导致生成重复的规则，进而破坏 kube-proxy 的功能
# nftables 与 kubeadm 不兼容:它会导致重复的防火墙规则和breaks kube-proxy
# nftables：是 Linux 防火墙子系统的框架（从 Linux 3.13 开始引入），它用于替代旧的iptables/ip6tables/arptables /ebtables 等工具


# 关闭selinux
# 为了绕过 SELinux 严格的访问控制机制，避免因其导致的各种奇怪权限问题。
# 根本原因：SELinux 会为每个进程和文件打上安全标签，限制其访问权限。容器运行时和 Kubernetes 组件（如 kubelet）需要访问宿主机文件系统，而 SELinux 的默认策略可能阻止这些操作，导致 Pod 网络故障、容器无法启动等问题。
#复杂性：要为 Kubernetes 正确配置 SELinux 策略非常复杂，大多数教程选择直接关闭以避免麻烦。

# centos
# 临时关闭 setenforce 0 
# 永久关闭 sed -i 's/enforcing/disabled/' /etc/selinux/config  
# 在 Ubuntu 系统中，SELinux 默认是未安装且未启用的
# 查看selinux的状态
sestatus

# 关闭swap
# Kubernetes 官方文档的强制性要求
# Kubelet（Kubernetes 的节点代理）在设计上要求精确管理资源，其默认行为是如果检测到 Swap 被启用，就会启动失败
# 性能考量：Swap 使用硬盘作为虚拟内存，速度远慢于物理内存。一旦启用，Pod 的性能会急剧下降。同时，Swap 会让调度器误判节点可用资源，导致 Pod 调度出错。
# 运维理念：Kubernetes 的哲学是“快速失败，快速恢复”。当 Pod 内存不足时，更希望它被直接杀死（OOM）并自动重启，而不是靠 Swap “续命”，导致节点响应缓慢，问题更难排查
# 查看当前 Swap 状态，此命令会列出所有激活的 Swap 分区或文件
sudo swapon --show
# 临时关闭swap，-a 参数代表关闭所有已知的 Swap 设备
sudo swapoff -a
# 永久关闭，但不会立刻关闭当前正在运行的 swap
sudo sed -ri 's/.*swap.*/#&/' /etc/fstab


# 在多个主机中修改hostname
sudo hostnamectl set-hostname master01
sudo hostnamectl set-hostname master02
sudo hostnamectl set-hostname worker01
sudo hostnamectl set-hostname worker02


# 在master添加hosts
# 作用：免记 IP，直接用名字互访，ping 10.0.0.3 可以修改为ping master01
cat >> /etc/hosts << EOF
10.0.0.3 master01
10.0.0.8 master02
10.0.0.6 worker01
10.0.0.17 worker02
EOF

# centos br_netfilter 这个内核模块默认自动加载
# 查看当前系统是否加载了br_netfilter
lsmod | grep br_netfilter
br_netfilter           32768  0
bridge                425984  1 br_netfilter
# 加载 br_netfilter 内核模块（Ubuntu 必须做这一步）
sudo modprobe br_netfilter
# 确保开机自动加载（写入 /etc/modules-load.d/）
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
br_netfilter
EOF

# 网络设置
# 将桥接的IPv4流量传递到iptables的链
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.ipv4.ip_forward = 1
EOF
# 核心作用：让 Kubernetes 的 kube-proxy（基于 iptables 模式）能够正确处理 容器网桥（Bridge） 上的网络包。
# 具体场景：你的 Pod 是跑在虚拟网卡（如 cni0、docker0）上的。当流量从 Pod 发出，经过这个“网桥”去往外部时，内核必须检查 iptables 规则（做 SNAT 地址转换），否则 Pod 无法访问外网，Service 的 ClusterIP 负载均衡也会失效。
# net.ipv4.ip_forward=1 是让节点间 Pod 网络互通的关键

# 立即生效（不需要重启）
sudo sysctl --system

# 时间同步
# 时间同步对Kubernetes集群至关重要：
# 证书验证：K8s组件间使用数字证书进行加密通信，证书有严格的有效期。节点间时间差过大会导致证书被认为“尚未生效”或“已过期”，造成kubelet无法连接apiserver等严重问题。
# 日志审计：集群审计日志和事件顺序依赖准确的时间戳，时间不同步会让排错变得异常困难

# ubuntu中，使用内置的 systemd-timesyncd 服务
# 检查当前的状态
timedatectl status
               Local time: Wed 2026-09-02 12:51:35 CST
           Universal time: Wed 2026-09-02 04:51:35 UTC
                 RTC time: Wed 2026-09-02 04:51:35
                Time zone: Asia/Shanghai (CST, +0800)
System clock synchronized: yes
              NTP service: active
          RTC in local TZ: no
# 如果上一步显示未开启，则重新开启
sudo timedatectl set-ntp true
```

### 1.1.2 安装docker

#### kubernetes与docker的版本兼容性

[docker的版本](https://docs.docker.com/engine/release-notes)

Kubernetes 1.24 及以上版本不再支持 Docker 作为 CRI（Container Runtime Interface），建议使用 containerd 或 CRI-O。

docker与containerd

- 当你执行docker run 时，典型调用链：Docker CLI → Docker Daemon (dockerd) → **containerd** → **runc**
- Containerd 是这个链条中的核心一环，它只专注于容器的核心生命周期管理，如创建、启动、停止容器，以及镜像的拉取和存储

针对 Kubernetes 1.24 及之后版本，关于 Docker 你需要注意的核心是：**Kubernetes 节点用于运行容器的“引擎”变了，但你构建的 Docker 镜像依然可以正常工作。**

通过K8s管理Pod下的容器不再通过docker管理，而直接通过containerd管理。真正受影响的是当你**直接登录到 K8s 的 Node 节点**上进行故障排查或维护时，`docker` 命令将无法看到k8s管理的容器。

在 containerd 环境下，你需要使用 **`ctr`** 作为主要的命令行工具来替代 `docker` 命令。

**`crictl`** 是 **Kubernetes 社区的调试工具**，专门用于管理和排查 Kubernetes 集群节点的容器问题

- **依赖与定位**：`ctr` 与 `containerd` 捆绑，安装 `containerd` 即有；`crictl` 需单独安装，通常在 K8s 节点上使用。`ctr -v` 显示 `containerd` 版本，`crictl -v` 显示 K8s 版本。
- **设计哲学**：`ctr` 优先保证功能完整性和底层访问能力；`crictl` 为 K8s 运维提供简洁、标准化的操作体验。
- **命名空间 (Namespace)**：`containerd` 用命名空间隔离资源。`crictl` 默认操作 `k8s.io` 命名空间；而 `ctr` 若不指定 `-n` 参数，则操作 `default` 命名空间。因此，用 `ctr` 查看 K8s 容器需加 `-n k8s.io`。

| 功能分类     | 原 Docker 命令       | crictl                  |
| :----------- | :------------------- | :---------------------- |
| **镜像管理** | `docker images`      | `crictl images`         |
|              | `docker pull`        | `crictl pull`           |
|              | `docker rmi`         | `crictl rmi`            |
| **容器管理** | `docker ps`          | `crictl ps`             |
|              | `docker exec`        | `crictl exec`           |
|              | `docker logs`        | `crictl logs`           |
|              | `docker stop` / `rm` | `crictl stop` / `rm`    |
| **Pod 管理** | (无直接命令)         | `crictl pods`           |
|              | (无直接命令)         | `crictl runp` / `stopp` |



单独安装crictl

```bash
# 下载安装包：https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.36.0
# crictl-v1.36.0-linux-amd64.tar.gz，这个压缩包里就一个crictl命令工具
sudo tar zxvf crictl-v1.36.0-linux-amd64.tar.gz -C /usr/local/bin/
```



#### [containerd](https://containerd.io/)

所以容器这一侧，只需要安装containerd就行，但你也可以直接安装最新的docker（它也依赖containerd）

k8s 1.36版本开始， kubelet将直接拒绝连接containerd 1.x，需要containerd 2.x的支持

[K8S的版本与containerd的版本支持情况](https://containerd.io/releases/#kubernetes-support)

- containerd：选用2.3.4
- K8S：选用1.36.4

```bash
# 我直接安装的docker，里面依赖的是containerd v2.3.4
sudo docker version
Client: Docker Engine - Community
 Version:           29.7.2
 API version:       1.55
 Go version:        go1.26.5
 Git commit:        a7dcaa6
 Built:             Wed Aug  5 18:28:53 2026
 OS/Arch:           linux/amd64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          29.7.2
  API version:      1.55 (minimum version 1.40)
  Go version:       go1.26.5
  Git commit:       6a43e3d
  Built:            Wed Aug  5 18:28:53 2026
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v2.3.4
  GitCommit:        db8809540e1a7a9da5d518876894933ff55692ab
 runc:
  Version:          1.4.3
  GitCommit:        v1.4.3-0-gbb14dabe
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0

containerd --version
containerd containerd v2.3.4 db8809540e1a7a9da5d518876894933ff55692ab


# 生成containerd默认配置文件（主配置文件）
sudo containerd config default | sudo tee /etc/containerd/config.toml

sudo vim /etc/containerd/config.toml

# 修改SystemdCgroup 为true
# 确保容器的 cgroup 驱动与 kubelet 保持一致（均为 systemd）
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runc.options]
  SystemdCgroup = true

# 检查SystemdCgroup是否配置成功
containerd config dump | grep -n -A 12 -B 4 'SystemdCgroup'

# 依情况配置containerd 的镜像站
# containerd 1.x 通常这样配置镜像站
# 在 [plugins."io.containerd.grpc.v1.cri"] 部分下，补充 registry.mirrors 配置（若已有该节点，直接添加内容）
[plugins."io.containerd.grpc.v1.cri".registry]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
      [plugins."io.containerd.grpc.v1.cri".registry.mirrors."docker.io"]
        endpoint = [
          "https://docker.1panel.live",
          "https://docker.1ms.run",
          "https://dytt.online"
        ]
 
# 但在containerd 2.x就不像在containerd 1.x这样了，1.x 是在 config.toml堆砌镜像配置了, 2.x会在/etc/containerd/certs.d 中
# 参考：https://blog.csdn.net/weimeilayer/article/details/162054757
# 查看指定的镜像配置位置
cat /etc/containerd/config.toml | grep -n -A 4 -B 4 config_path

27-    [plugins.'io.containerd.cri.v1.images'.pinned_images]
28-      sandbox = 'registry.k8s.io/pause:3.10.2'
29-
30-    [plugins.'io.containerd.cri.v1.images'.registry]
31:      config_path = ''
32-
33-    [plugins.'io.containerd.cri.v1.images'.image_decryption]
34-      key_model = 'node'
35-
--
150-  [plugins.'io.containerd.nri.v1.nri']
151-    disable = false
152-    socket_path = '/var/run/nri/nri.sock'
153-    plugin_path = '/opt/nri/plugins'
154:    plugin_config_path = '/etc/nri/conf.d'
155-    plugin_registration_timeout = '5s'
156-    plugin_request_timeout = '2s'
157-    disable_connections = false
158-
--
257-    max_concurrent_downloads = 3
258-    concurrent_layer_fetch_buffer = 0
259-    max_concurrent_uploaded_layers = 3
260-    check_platform_supported = false
261:    config_path = ''
262-    max_concurrent_unpacks = 1
263-
264-[cgroup]
265-  path = ''

# 修改第31行的config_path
config_path='/etc/containerd/certs.d'

# 配置docker.io的镜像站
sudo mkdir -p /etc/containerd/certs.d/docker.io
sudo vim /etc/containerd/certs.d/docker.io/hosts.toml
# 解释
# 参数	说明
# server	默认的镜像仓库地址，回退使用
# host."<url>"	镜像加速器地址，按顺序尝试
# capabilities	该加速器支持的能力：pull（拉取）、resolve（解析）、push（推送），多个 host	Containerd 会按配置顺序依次尝试，直到成功

server = "https://docker.io"

[host."https://mirror.ccs.tencentyun.com"]
  capabilities = ["pull", "resolve"]
[host."https://dockerpull.cn"]
  capabilities = ["pull", "resolve"]
 
# 除了配置docker.io的，我们要配置k8s的
sudo mkdir -p /etc/containerd/certs.d/registry.k8s.io
sudo vim /etc/containerd/certs.d/registry.k8s.io/hosts.toml
server = "https://registry.k8s.io"

[host."https://mirror.ccs.tencentyun.com"]
   capabilities = ["pull", "resolve"]
  
# 如果你的镜像来源于其他io，例如：ghcr.io等
# 你就要新建一个 sudo mkdir -p /etc/containerd/certs.d/ghcr.io，然后如上操作

sudo systemctl restart containerd
systemctl enable containerd
systemctl status containerd
```

### 1.1.3 安装k8s 套件

[kubernetes的版本](https://github.com/kubernetes/kubernetes/tree/master/CHANGELOG)，本次K8S：选用1.36.4

| 压缩包                                 | 部署目标节点                        | 二进制                                                       |
| -------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| `kubernetes‑client‑linux‑amd64.tar.gz` | **运维管理机（不跑 k8s 服务）**     | `kubectl`、`kubectl‑convert`（仅 2 个）                      |
| `kubernetes‑node‑linux‑amd64.tar.gz`   | **仅 Worker 工作节点**              | `kubeadm`、`kubectl`、`kubectl‑convert`、`kubelet`、`kube‑log‑runner`、`kube‑proxy`、`mounter` |
| `kubernetes‑server‑linux‑amd64.tar.gz` | **Master 控制平面节点（全集大包）** | 全套控制平面组件 + node 全部二进制 + `.tar`镜像包 + `.docker_tag`版本文件 |

**master 节点 和 worker 节点，两个都可以使用 `kubernetes‑server‑linux‑amd64.tar.gz`**。 server 包是**完整全集包**：里面既有控制平面组件，也包含全部 node 节点二进制。

`server/bin` 下的tar包，k8s 用 kubeadm 部署集群时： 控制平面（apiserver、controller‑manager、scheduler）、kube‑proxy，**是以容器方式运行在 containerd/docker 里面**，不是直接跑系统二进制。 `.tar`就是官方预先导出好的容器镜像文件，专门用于**离线隔离环境（在无外网的情况下，直接可以使用这些镜像）**。

关于那些镜像tar包：

- 有外网环境（可以访问 registry.k8s.io）：不需要手动导入 tar 包
  - 执行 `kubeadm init` / `kubeadm join` 的时候，kubeadm 会自动从官方镜像仓库拉取需要的容器镜像。 server 包里所有 `.tar` 文件完全闲置，不用处理。
- 离线环境（服务器不能访问外网，无法拉镜像）：**必须手动导入到 containerd /docker，否则 kubeadm init 会报错镜像拉取失败**
  - kubeadm **不会自动读取本地目录下的 tar 文件**！不会自动加载，必须你手动执行导入命令。 kubeadm 只去容器运行时（containerd）的镜像仓库里找镜像，磁盘上的 tar 文件它不认。

K8s 1.24 及以上，**默认使用 containerd，tar 镜像只能导入 containerd；除非额外部署 cri‑dockerd，否则 docker load 导入镜像对 k8s 集群无效**。

#### 套件

k8s中，kubernetes‑server‑linux‑amd64.tar.gz解压后，/kubernetes/server/bin文件夹下的这些工具和tar镜像包：

**1. 核心控制平面组件（二进制）**

- **`kube-apiserver`**：Kubernetes 集群的**网关**，所有 API 请求的入口，负责认证、授权、校验和 RESTful API 服务。
- **`kube-controller-manager`**：运行各种**控制器**（如节点控制器、副本控制器）的进程，负责将集群状态调谐至期望状态。
- **`kube-scheduler`**：**调度器**，负责为新创建的 Pod 选择最优的合适节点运行。
- **`kube-aggregator`**：**API 聚合层服务**，允许将其他扩展 API（如 metrics-server）集成到主 API 路径下，实现 API 的横向扩展。
- **`apiextensions-apiserver`**：处理 **CRD（自定义资源定义）** 的请求，让用户能扩展 Kubernetes 原生资源类型。

**2. 节点与运维组件（二进制）**

- **`kubelet`**：运行在**每个工作节点**上的核心代理，负责管理 Pod 及其容器的生命周期（拉取镜像、启动容器等）。
- **`kube-proxy`**：运行在**每个节点**上的网络代理，负责维护网络规则（如 iptables/IPVS），实现 Service 的负载均衡和集群内服务发现。
- **`kubeadm`**：集群**安装引导工具**，用于快速初始化集群（`kubeadm init`）、加入节点（`kubeadm join`）和升级集群。

**3. 命令行与辅助工具（二进制）**

- **`kubectl`**：与 API Server 通信的**命令行工具**，用于增删改查集群资源（如 `get pods`）。
- **`kubectl-convert`**：清单转换工具，用于在不同 API 版本之间转换 YAML/JSON 配置文件（如从 `v1beta1` 转 `v1`）。
- **`kube-log-runner`**：**日志转发辅助进程**，通常在 systemd 单元文件中配合 kubelet 使用，用于捕获并重定向日志。
- **`mounter`**：**挂载辅助工具**，用于执行特权挂载操作（如挂载 NFS、CIFS 等外部存储卷），解决 kubelet 挂载时的权限或环境问题。

**4. 容器镜像包（`.tar` 及 `.docker_tag`）**

- **`kube-apiserver.tar` / `kube-controller-manager.tar` / `kube-scheduler.tar` / `kube-proxy.tar`**：分别是上述核心组件的**容器镜像离线包**。在离线安装时，需用 `docker load` 或 `ctr image import` 将其导入节点本地容器运行时。
- **`kube-apiserver.docker_tag` 等文件**：纯文本文件，内部只包含对应镜像的 **Tag 版本号**（如 `v1.28.0`）。主要用于部分安装脚本读取，以确定加载镜像时的正确版本标签。

**核心区别**：`kube-apiserver`（无后缀）是直接运行的**二进制进程**（通常以 Static Pod 运行），而 `kube-apiserver.tar` 是把该进程打包成的**容器镜像离线包**（用于在容器化环境中运行该组件）。

在标准的 kubeadm 安装中，控制平面组件通常以容器方式运行，因此会用到 `.tar` 包；使用 kubeadm 对 k8s 集群进行初始化时，所有的集群组件都将以容器的方式运行的，因此要准备 k8s 集群的容器镜像。（由于使用 kubeadm 部署集群，集群所有核心组件均以 Pod 运行，需要为主机准备镜像，不同角色主机准备不同镜像。）

而 `kubectl`、`kubelet`、`kubeadm` 则必须在宿主机上作为二进制文件安装。



```bash
# 解压
tar -zxvf kubernetes-server-linux-amd64.tar.gz

cd kubernetes/server/bin

sudo mv apiextensions-apiserver kubeadm kube-aggregator kube-apiserver kube-controller-manager kubectl kubectl-convert kubelet kube-log-runner kube-proxy kube-scheduler mounter /usr/local/bin/

# 查看 控制平面 需要的镜像
kubeadm config images list
registry.k8s.io/kube-apiserver:v1.36.4
registry.k8s.io/kube-controller-manager:v1.36.4
registry.k8s.io/kube-scheduler:v1.36.4
registry.k8s.io/kube-proxy:v1.36.4
registry.k8s.io/coredns/coredns:v1.14.2
registry.k8s.io/pause:3.10.2
registry.k8s.io/etcd:3.6.8-0

kubeadm config images list --kubernetes-version=v1.36.4 --image-repository mirror.ccs.tencentyun.com
# `config images list` 专门用于列出（list）部署 Kubernetes 控制平面所需的所有容器镜像列表。它只做“查询”和“展示”，不会真的下载镜像。
# `--kubernetes-version=v1.36.4` 指定目标 Kubernetes 的版本号。kubeadm 会根据这个版本计算出需要哪些组件（如 apiserver、controller-manager 等）以及它们的对应版本标签。
# `--image-repository mirror.ccs.tencentyun.com`：指定镜像仓库地址。Kubernetes 默认的官方仓库是 registry.k8s.io（旧版为 k8s.gcr.io），在国内访问速度较慢。加上这个参数后，所有镜像名会被强制加上 mirror.ccs.tencentyun.com 前缀，从而利用腾讯云的国内加速镜像（仅在腾讯云内网可以加速）。
mirror.ccs.tencentyun.com/kube-apiserver:v1.36.4
mirror.ccs.tencentyun.com/kube-controller-manager:v1.36.4
mirror.ccs.tencentyun.com/kube-scheduler:v1.36.4
mirror.ccs.tencentyun.com/kube-proxy:v1.36.4
mirror.ccs.tencentyun.com/coredns:v1.14.2
mirror.ccs.tencentyun.com/pause:3.10.2
mirror.ccs.tencentyun.com/etcd:3.6.8-0

cd kubernetes/server/bin
# 导入控制平面组件镜像，下面的这几个是kubernetes-server-linux-amd64.tar.gz中有的
# kubeadm init 本身也可以直接拉镜像，参数 `--image-repository=mirror.ccs.tencentyun.com`，它会自己去拉全部镜像（apiserver 等也会拉）。
sudo ctr -n k8s.io images import kube-apiserver.tar
sudo ctr -n k8s.io images import kube-controller-manager.tar
sudo ctr -n k8s.io images import kube-scheduler.tar
sudo ctr -n k8s.io images import kube-proxy.tar
#ctr：这是 containerd 自带的客户端命令行工具，用于和 containerd 守护进程交互。它提供了比 docker 更底层的、面向调试和管理员的操作接口。

#-n k8s.io：-n 是 --namespace 的缩写，指定了操作所在的命名空间 (namespace)。
# containerd 使用命名空间来隔离资源，让不同用途的容器和镜像互不干扰。
# k8s.io 是 Kubernetes 专用的命名空间。Kubernetes 管理的所有容器和镜像都默认存储在这里。
# 如果你不加 -n k8s.io，ctr 会默认使用 default 命名空间，这样 Kubernetes 就无法识别你导入的镜像。

# images import：这是 ctr 的镜像管理子命令，用于导入镜像。
# import 的功能是从一个 .tar 归档文件中加载镜像，并将其存储到 containerd 的本地镜像库中。
# kube-apiserver.tar：这是待导入的镜像文件。它通常是通过 docker save 或 ctr image export 命令将镜像保存而成的。
# 如果配置的containerd的镜像站之后，通过下面的命令，无法安装 那么采用其他办法
ctr -n k8s.io images pull registry.k8s.io/coredns/coredns:v1.14.2
ctr -n k8s.io images pull registry.k8s.io/pause:3.10.2
ctr -n k8s.io images pull registry.k8s.io/etcd:3.6.8-0

# 检索：https://docker.aityp.com/
# 使用其他镜像站的地址，然后通过打tag的方式，将他和register.k8s.io建立关系
sudo ctr -n k8s.io images pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/coredns/coredns:v1.14.2
sudo ctr -n k8s.io images tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/coredns/coredns:v1.14.2 registry.k8s.io/coredns/coredns:v1.14.2

sudo ctr -n k8s.io images pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/pause:3.10.2
sudo ctr -n k8s.io images tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/pause:3.10.2 registry.k8s.io/pause:3.10.2

sudo ctr -n k8s.io images pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/etcd:3.6.8-0
sudo ctr -n k8s.io images tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/registry.k8s.io/etcd:3.6.8-0 registry.k8s.io/etcd:3.6.8-0
```

### 1.1.4 添加证书

[添加证书官方说明](https://kubernetes.io/docs/setup/best-practices/certificates/)

kube-apiserver 和 etcd 之间的通信必须通过 TLS 加密，所以需要证书。

在 PKI 体系中，**根 CA 是信任链的顶端锚点，中间 CA 是由根 CA 签发的下级签发机构，企业 CA 是在企业内部自建的 CA 体系（可以是根也可以是中间），自签名 CA 则是未经上级签发、自己给自己签名的证书**。

证书的来源是灵活的：

1. **自签名证书**：kubeadm 默认在 `kubeadm init` 时自动生成一套完整的自签 CA 和证书，存放在 `/etc/kubernetes/pki` 目录
2. **外部 CA**：你可以预先在 `--cert-dir` 指定的目录（默认 `/etc/kubernetes/pki`）放置自己的证书和密钥，kubeadm 检测到已存在的证书对时不会覆盖
3. **企业 CA / 中间 CA**：kubeadm 支持两级 CA 架构，企业部署中通常使用外部 CA 来签发证书
4. **公共 CA**：理论上也可以用于 API server 的服务端证书（如 Let’s Encrypt），但内部组件认证一般仍需私有 CA



## 1.1 kubeadm方式

kubeadm是官方社区推出的一个用于快速部署kubernetes集群的工具。

```bash
# 创建一个 Master 节点
kubeadm init

# 将一个 Node 节点加入到当前集群中
kubeadm join <Master节点的IP和端口 >
```







# 2 k8s核心概念



# 3 搭建集群监控平台



# 4 高可用k8s集群



# 5 集群部署项目





# 其他

| 特性维度         | AppArmor                                                     | SELinux                                                      |
| :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **控制模型**     | **基于路径 (Path-based)** 为特定程序设置能访问哪些文件路径的规则。 | **基于标签 (Label-based)** 给系统里的每个文件、进程、端口都打上标签，根据标签之间的规则决定访问权限。 |
| **策略形式**     | 人类可读的**文本配置文件** 规则一目了然，修改方便。          | 需编译的**二进制策略模块** 规则策略需要编译后才能加载，管理更复杂。 |
| **默认发行版**   | **Ubuntu**、openSUSE                                         | **RHEL**、Fedora、CentOS                                     |
| **文件系统依赖** | **无特殊要求**                                               | **需要支持扩展属性 (xattrs)**，如 ext4 标签是作为文件的扩展属性存储的。 |
| **性能影响**     | **较小** 基于路径的字符串匹配，开销相对低。                  | **略高** 需要维护和查询每个文件/进程的标签。                 |
| **学习与运维**   | **学习曲线平缓** 使用 `aa-status`、`aa-logprof` 等工具，上手快。 | **学习曲线陡峭** 概念复杂（类型、角色、用户等），需理解 `audit2allow` 等高级工具。 |
| **高级安全模型** | 不支持 MLS/MCS                                               | **原生支持** MLS (多级安全) 和 MCS (多类别安全)              |

