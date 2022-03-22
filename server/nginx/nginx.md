# Nginx

# 1 基本概念

nginx是一个高性能的HTTP和反向代理服务器，特点是占有内存少，并发能力强（为性能优化而开发的）。

nginx可以作为静态页面的服务器，同时支持CGI协议的动态语言（eg：PHP），但不支持Java（只能通过tomcat配合完成）

## 1.1 反向代理

<img src="./figure/正向代理与反向代理.png" style="zoom: 67%;" />

### 正向代理

1. 客户端**需要配置**代理服务器，客户端向代理服务器发送请求，并指定目标服务器地址。
2. 然后由代理服务器和原始服务器通信，转交请求，再返回给客户端。
3. 正向代理隐藏了真实的客户端，使真实客户端对服务器不可见。

### 反向代理

1. 客户端对代理无感知，客户端**无需做任何配置**就可以访问。
2. 我们只需要将请求发送到反向代理服务器，由反向代理服务器去选择目标服务器。
3. 获取数据后，再返回给客户端。
4. 此时反向代理服务器和目标服务器对外就是一个服务器，暴露的是代理服务器地址，隐藏了真实服务器IP地址。
5. 反向代理隐藏了真实的服务器，使真实服务器对客户端不可见。

## 1.2 负载均衡

例如淘宝双十一的商品秒杀，单个服务器、或者增加了内存、cpu的单个服务器已经无法在短时承载如此多的并发请求，所以原先请求集中到单个服务器上的情况，改为将请求分发到多个服务器上。

通过nginx代理服务器，把来自客户端的并发请求均匀分配给真实服务器集群。——负载均衡

# 2 编译及部署

## 2.1 编译

### 2.1.1 安装编译工具及依赖库

在安装nginx之前需要装编译工具，和nginx的依赖库等等。

```bash
yum -y install gcc pcre-devel zlib-devel openssl-devel libxml2-devel \
libxslt-devel gd-devel GeoIP-devel jemalloc-devel libatomic_ops-devel \
perl-devel per-ExtUtils-Embed

# yum -y install 与 yum install 的区别是：在安装过程中如有询问，自动选择y（yes），而不需要手动选择。

# 在安装依赖时，libatomic_ops-devel，per-ExtUtils-Embed 这两个工具安装找不到，但好像也不影响接下来的安装，暂且不安装吧。

# 尚硅谷的在视频中它安装了gcc，zlib zlib-devel pcre-devel openssl openssl-devel

```

### 2.1.2 nginx源码（安装包）获取及安装

```bash
mkdir -p /opt/data/source
cd /opt/data/source
wget http://nginx.org/download/nginx-1.17.4.tar.gz
tar zxmf nginx-1.17.4.tar.gz
```

编译nginx源码文件时，首先需要通过编译配置命令configure进行编译配置。

很多安装包解压后，文件中都有一个configure命令，它用于进行安装的编译配置。

在nginx中，编译配置命令configure的常用编译配置可查询互联网，以下仅列举部分配置参数：

| 编译配置参数                     | 默认值/默认编译状态 | 参数说明               |
| -------------------------------- | ------------------- | ---------------------- |
| --prefix=PATH                    | /usr/local          | 编译后代码的安装目录   |
| --with-threads                   | 不编译              | 启用线程池支持         |
| --without-http_auth_basic_module | 编译                | 不编译http基本认证模块 |

“---with“前缀的编译配置参数的模块都不会被默认编译，而”--without“前缀都会被默认编译。

```bash
# 进入解压后的安装文件夹
cd nginx-1.17.4
# 代码编译 && 编译后安装
make && make install
```

### 2.1.3 添加第三方模块

nginx的功能是以模块的方式存在的，同时也支持添加第三方开发的功能模块。执行configure时，通过--add-module=PATH参数指定第三方模块的代码路径，然后再编译安装。

```bash
# 添加第三方静态模块的方法如下：
./configure --add-module=../ngx_http_proxy_connect_module
# 添加第三方动态模块的方法如下：
./configure --add-dynamic-module=../ngx_http_proxy_connect_module --with-compat

make && make install
```

## 2.2 部署

