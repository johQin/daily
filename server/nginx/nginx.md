# Nginx

# 实践

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

## 1.3 动静分离

![](./figure/动静分离.png)

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
# 安装成功后，会多出一个usr/local/nginx文件夹，
# 在此文件夹下的sbin下有启动脚本
```

### 2.1.3 启动nginx服务

安装之后，我们需要启动nginx服务，

```bash
# 进入含有启动脚本的文件夹
cd /usr/local/nginx/sbin
# 执行启动脚本
./nginx
# 查看是否已启动相应的进程
ps -ef | grep nginx

root     2624670       1  0 17:54 ?        00:00:00 nginx: master process ./nginx
nobody   2624671 2624670  0 17:54 ?        00:00:00 nginx: worker process
root     2626322 2571514  0 17:55 pts/0    00:00:00 grep --color=auto nginx
# 证明我们已经完成了nginx的启动

pwd
cd /usr/local/nignx
cd conf
vim nginx.conf

# server {
#        listen       80;
#        server_name  localhost;
#
#        #charset koi8-r;
#
#        #access_log  logs/host.access.log  main;
#
#       location / {
#            root   html;
#            index  index.html index.htm;

# 我们可以看到nginx的服务器监听的是80端口
# 访问ip + 端口号 它会跳出nginx页面
```

![](./figure/nginx启动后.png)

如果你是购买的云服务器，可能系统默认会有防火墙，你需要在它提供的控制台中更改防火墙设置。

如果对80端口有设置防火墙，那么你就无法访问此nginx服务器，所以需要查看防火墙的情况。

```bash
# 查看防火墙开放的端口号
firewall-cmd --list-all
# 如果出现FirewallD is not running，说明防火墙服务处于关闭状态。
# 查看防火墙状态
systemctl status firewalld

● firewalld.service - firewalld - dynamic firewall daemon
   Loaded: loaded (/usr/lib/systemd/system/firewalld.service; disabled; vendor preset: enabled)
   Active: inactive (dead)# 果然处于关闭状态。
     Docs: man:firewalld(1)
# 开启防火墙，没有任何提示即开启成功
systemctl start firewalld # 关闭防火墙 systemctl stop firewalld

# 开放端口
firewall-cmd --add-port=80/tcp --permanant # --permanent永久生效，没有此参数重启后失效
# 开放后需要重启防火墙，即可生效。
firewall-cmd -reload

# 关闭端口
firewall-cmd --zone= public --remove-port=80/tcp --permanent #--zone 作用域
```



### 2.1.4 添加第三方模块

nginx的功能是以模块的方式存在的，同时也支持添加第三方开发的功能模块。执行configure时，通过--add-module=PATH参数指定第三方模块的代码路径，然后再编译安装。

```bash
# 添加第三方静态模块的方法如下：
./configure --add-module=../ngx_http_proxy_connect_module
# 添加第三方动态模块的方法如下：
./configure --add-dynamic-module=../ngx_http_proxy_connect_module --with-compat

make && make install
```

## 2.2 部署

# 3 配置指令

nginx的配置文件所在的位置：`/usr/local/nginx/conf/nginx.conf`

```bash
#user  nobody;
worker_processes  1;

#error_log  logs/error.log;
#error_log  logs/error.log  notice;
#error_log  logs/error.log  info;

#pid        logs/nginx.pid;


events {
    worker_connections  1024;
}


http {
    include       mime.types;
    default_type  application/octet-stream;

    #log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
    #                  '$status $body_bytes_sent "$http_referer" '
    #                  '"$http_user_agent" "$http_x_forwarded_for"';

    #access_log  logs/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    #keepalive_timeout  0;
    keepalive_timeout  65;

    #gzip  on;

    server {
        listen       80;
        server_name  localhost;

        #charset koi8-r;

        #access_log  logs/host.access.log  main;

        location / {
            root   html;
            index  index.html index.htm;
        }

        #error_page  404              /404.html;

        # redirect server error pages to the static page /50x.html
        #
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }

        # proxy the PHP scripts to Apache listening on 127.0.0.1:80
        #
        #location ~ \.php$ {
        #    proxy_pass   http://127.0.0.1;
        #}

        # pass the PHP scripts to FastCGI server listening on 127.0.0.1:9000
        #
        #location ~ \.php$ {
        #    root           html;
        #    fastcgi_pass   127.0.0.1:9000;
        #    fastcgi_index  index.php;
        #    fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
        #    include        fastcgi_params;
        #}

        # deny access to .htaccess files, if Apache's document root
        # concurs with nginx's one
        #
        #location ~ /\.ht {
        #    deny  all;
        #}
    }


    # another virtual host using mix of IP-, name-, and port-based configuration
    #
    #server {
    #    listen       8000;
    #    listen       somename:8080;
    #    server_name  somename  alias  another.alias;

    #    location / {
    #        root   html;
    #        index  index.html index.htm;
    #    }
    #}


    # HTTPS server
    #
    #server {
    #    listen       443 ssl;
    #    server_name  localhost;

    #    ssl_certificate      cert.pem;
    #    ssl_certificate_key  cert.key;

    #    ssl_session_cache    shared:SSL:1m;
    #    ssl_session_timeout  5m;

    #    ssl_ciphers  HIGH:!aNULL:!MD5;
    #    ssl_prefer_server_ciphers  on;

    #    location / {
    #        root   html;
    #        index  index.html index.htm;
    #    }
    #}

}

```



配置指令（配置项）就在配置文件中。

## 3.1 常用命令

使用nginx命令的前提条件：必须进入`usr/local/nginx/sbin`，否则就要配置环境命令。

```bash
# 1.查看nginx版本号
./nginx -v

# 2.启动nginx命令
./nginx

ps -ef | grep nginx

root     2624670       1  0 17:54 ?        00:00:00 nginx: master process ./nginx
nobody   2624671 2624670  0 17:54 ?        00:00:00 nginx: worker process
root     2626322 2571514  0 17:55 pts/0    00:00:00 grep --color=auto nginx

# 3.关闭nginx命令
./nginx -s stop

ps -ef | grep nginx
root     2992902 2571514  0 21:18 pts/0    00:00:00 grep --color=auto nginx

# 4.重新加载nginx，而不是重启服务器
./nginx -s reload
# 当我们修改了/usr/local/nginx/conf下的 nginx.conf后，如果想要这些配置生效那么就需要重新加载。


```

## 3.2 约定名词

- 配置指令（directive）
  - 由nginx约定的内部固定字符串
  - nginx的每个功能配置都是通过多个不同的指定组合来实现的
- 配置指令值
  - 配置指令对应的值，指令与指令值是一对多的关系。
- 配置指令语句
  - 由指令与指令值构成指令语句，语句结束需要用`;`作为结束
- 配置指令域
  - 配置指令值有时会由`{}`括起来的指令语句集合。本书中约定`{}`括起来的部分为配置指令域，简称指令域。
- 配置全局域：配置文件`ginx.conf`中上层没有其他指令域的区域被称为配置全局域，简称全局域。

| 域名称   | 域类型 | 域说明                                                       |
| -------- | ------ | ------------------------------------------------------------ |
| main     | 全局域 | nginx的根级别指令区域，该区域的配置指令是全局有效的，该指令名为**隐性显示**，nginx.conf的整个文件内容都写在该指令域中。 |
| events   | 指令域 | nginx事件驱动相关的配置指令域                                |
| http     | 指令域 | Http核心配置指令域，包含客户端完整http请求过程中，每个过程的处理方法的配置指令 |
| upstream | 指令域 | 用于定义被代理服务器组的指令区域，也称“上游服务器”           |
| server   | 指令域 | nginx用来定义服务IP、绑定端口及服务相关的指令区域            |
| location | 指令域 | 对用户URI进行访问路由处理的指令域                            |
| stream   | 指令域 | 对tcp协议实现代理的配置指令域                                |
| types    | 指令域 | 定义被请求文件扩展名与MIME类型映射表的指令区域               |
| if       | 指令域 | 按照选择条件判断为真时，使用的配置指令域                     |

在nginx.conf的最外层主要有三个部分（三个域）组成：

1. main全局域
2. events指令域
3. http指令域

三个域也需要其他指令域一起配合使用



## 3.2 main全局域

从配置文件开始到events块之间的内容，主要会设置一些影响nginx服务器整体运行的配置指令。

主要包括：nginx服务器的用户（组）、允许生成的worker process数、进程PID存放路径、日志存放路径和类型以及配置文件的引入等。

## 3.3 events指令域

涉及的指令主要影响nginx服务器与用户的网络连接。

常用的设置包括是否开启对多work process下的网络连接进行序列化，是否允许同时接收多个网络连接，选取哪种事件驱动模型来处理连接请求，每个worker process 可以同时支持的最大连接数等。

## 3.4 http指令域

是nginx服务器配置最为频繁的部分。

http自身域配置的指令包括文件引入，MIME-TYPE定义，日志自定义，连接超时时间，单连接请求数的上限等

http指令域，除自身域外，还包含了多个server指令域，而server指令域又包含了多个location指令域。

## 3.5 server指令域

与虚拟主机有密切联系，该技术的产生是为了节省互联网服务器硬件成本。

从用户的角度看，虚拟主机和一台独立的硬件主机是完全一样的。

最常见的配置是本虚拟主机的监听配置和本虚拟主机的名称或IP配置。

一个server指令域可配置多个location 指令域

## 3.6 location指令域

主要作用是对nginx服务器接收的请求地址url进行解析。

匹配url上除ip地址之外的字符串，对特定的请求进行处理。

例如：地址定向、数据缓存和应答控制等功能，还有许多第三方模块的配置也在这里进行配置。

URL（统一资源定位符）是URI（统一资源标识符）的子集。

URI，即统一标识资源符，通用的URI语法格式如下：

`scheme:[// [user[:password]@]  host  [:port] ]  [/path]  [?query]  [#fragment]`

### URI的匹配规则

`location [ = | ~ | ~* | ^~ | @ ] pattern {   }`

中括号中的是修饰语，pattern是匹配项

- 无修饰语：完全匹配url中除访问参数以外的内容，匹配项只能是字符串，不能是正则表达式
  - 
- 修饰语 `=` ：完全匹配url中除访问参数以外的内容，
  - Linux下会区分大小写，windows则不会
- 修饰语 `~` ：完全匹配url中除访问参数以外的内容，
  - Linux下会区分大小写，windows则不会，
  - 匹配项的内容必须是正则表达式
- 修饰语 `~*` ：完全匹配url中除访问参数以外的内容
  - 不区分大小写
  - 匹配项的内容必须是正则表达式
- 修饰语 `^~` ：完全匹配url中除访问参数以外的内容
  - 匹配项的内容如果不是正则表达式，则不再进行正则表达式测试
- 修饰语 `@` ：定义一个只能内部访问的location区域，可以被内部跳转指令使用

#### 匹配顺序

1. 先检测匹配项内容为非正则表达式修饰语修饰的location，后检测匹配项内容为正则表达式修饰语修饰的location
2. 当匹配项内容同时被非正则与正则都匹配的location，按匹配项内容为正则匹配location执行。
3. 所有匹配项内容均为非正则表达式的location，按照匹配项的内容完全匹配的内容长短进行匹配，即匹配内容多的location被执行
4. 所有匹配项的内容均为正则表达式的location，按照书写的先后顺序进行匹配，匹配后就执行，不再做后续检测。

#### proxy_pass

当location为正则匹配且内部有proxy_pass指令时，proxy_pass的指令值不能包含无变量的字符串，修饰语不受该规则限制。

```bash
location ~ /images {
	proxy_pass	http:127.0.0.1:8080;	#正确的指令值
	proxy_pass	http:127.0.0.1:8080$request_uri;	#正确的指令值
	proxy_pass	http:127.0.0.1:8080/image$request_uri;	#正确的指令值
	proxy_pass	http:127.0.0.1:8080/;	#错误的指令值
}
```



# 4 配置实例

准备工作：在linux系统中安装tomcat，使用默认端口8080。

## 4.1 准备工作

#### 安装jdk

```bash
# tomcat需要jdk作为环境，所以首先要安装jdk。下载之后默认的目录为： /usr/lib/jvm/
yum search java | grep jdk

openjdk-asmtools-javadoc.noarch : Javadoc for openjdk-asmtools
java-1.8.0-openjdk.x86_64 : OpenJDK 8 Runtime Environment
java-1.8.0-openjdk-accessibility.x86_64 : OpenJDK 8 accessibility connector
java-1.8.0-openjdk-demo.x86_64 : OpenJDK 8 Demos
java-1.8.0-openjdk-devel.x86_64 : OpenJDK 8 Development Environment
...
java-1.8.0-openjdk-src.x86_64 : OpenJDK 8 Source Bundle
java-11-openjdk.x86_64 : OpenJDK 11 Runtime Environment
...
java-11-openjdk-jmods.x86_64 : JMods for OpenJDK 11
java-11-openjdk-src.x86_64 : OpenJDK 11 Source Bundle
...

# 安装
yum install java-1.8.0-openjdk
# 我在腾讯云，安装完成后，java不需要配环境变量，就可以直接java -version
# 相关原因：
#ubuntu12.10系统使用ppa方式下载并自动安装jdk后，java被安装到usr/lib/jvm目录下，没有修改环境变量便可以使用。
#这是因为操作系统将java的可执行文件先做成链接放在了/etc/alternatives下，然后又把alternatives下的链接又做成了链接放在了/usr/bin下。
#alternative是可选项的意思
#首先，因为依赖关系的存在，一个软件包在系统里面可能出现新旧版本并存的情况，或者同时安装了多种桌面环境， 系统更新之后会自动将最后安装的版本作为默认值。
#在以前，要想用旧版本作为默认值就必须要手动修改配置文件，有些软件比较简单，有些却要修改很多文件，甚至一些相关软件的配置文件也要相应修改。
#update-alternatives命令就是为了解决这个问题的，指定一个简写的名称后会根据每个软件包的具体情况给出一些选项，自动完成一些配置文件的修改，减轻系统维护的负担。

# 因为yum 安装位置在/usr/lib/jvm/，进入jvm我们可以看见多个文件，里面包括一些软连接

[root@VM-4-8-centos jvm]# ll
总用量 4
# 第一列为文件类型，我们可以看到除了第一个文件的文件类型是d（文件夹），其余都是连接文件l（符号连接），而且还是软连接，因为后面有 “->” 符号，软连接，指向其他文件。真正的文件是箭头后面那个。
drwxr-xr-x 3 root root 4096 3月  28 21:15 java-1.8.0-openjdk-1.8.0.312.b07-2.el8_5.x86_64
lrwxrwxrwx 1 root root   21 3月  28 21:15 jre -> /etc/alternatives/jre
lrwxrwxrwx 1 root root   27 3月  28 21:15 jre-1.8.0 -> /etc/alternatives/jre_1.8.0
lrwxrwxrwx 1 root root   35 3月  28 21:15 jre-1.8.0-openjdk -> /etc/alternatives/jre_1.8.0_openjdk
lrwxrwxrwx 1 root root   51 11月 13 16:29 jre-1.8.0-openjdk-1.8.0.312.b07-2.el8_5.x86_64 -> java-1.8.0-openjdk-1.8.0.312.b07-2.el8_5.x86_64/jre
lrwxrwxrwx 1 root root   29 3月  28 21:15 jre-openjdk -> /etc/alternatives/jre_openjdk

```

#### 安装tomcat

```bash
 # 在/usr/local下，下载tomcat安装包
 wget https://dlcdn.apache.org/tomcat/tomcat-8/v8.5.77/bin/apache-tomcat-8.5.77.tar.gz
 # 解压安装包
 tar -xvf apache-tomcat-8.5.77.tar.gz
 cd /apache-tomcat-8.5.77/bin
 # 执行安装脚本
 ./startup.sh

Using CATALINA_BASE:   /usr/local/apache-tomcat-8.5.77
Using CATALINA_HOME:   /usr/local/apache-tomcat-8.5.77
Using CATALINA_TMPDIR: /usr/local/apache-tomcat-8.5.77/temp
Using JRE_HOME:        /usr
Using CLASSPATH:       /usr/local/apache-tomcat-8.5.77/bin/bootstrap.jar:/usr/local/apache-tomcat-8.5.77/bin/tomcat-juli.jar
Using CATALINA_OPTS:   
Tomcat started.

 # cd ../logs，查看tomcat的日志，如果能看到那么启动成功
 tail -f catalina.out
 
 28-Mar-2022 22:41:27.060 信息 [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory 把web 应用程序部署到目录 [/usr/local/apache-tomcat-8.5.77/webapps/examples]
28-Mar-2022 22:41:27.203 信息 [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Web应用程序目录[/usr/local/apache-tomcat-8.5.77/webapps/examples]的部署已在[143]毫秒内完成
28-Mar-2022 22:41:27.204 信息 [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory 把web 应用程序部署到目录 [/usr/local/apache-tomcat-8.5.77/webapps/docs]
28-Mar-2022 22:41:27.213 信息 [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory Web应用程序目录[/usr/local/apache-tomcat-8.5.77/webapps/docs]的部署已在[9]毫秒内完成
28-Mar-2022 22:41:27.214 信息 [localhost-startStop-1] org.apache.catalina.startup.HostConfig.deployDirectory 把web 应用程序部署到目录

 # tomcat默认8080端口，所以还需要开放一下8080端口的防火墙
```

![](./figure/tomcat服务器启动后.PNG)



## 4.2 反向代理：实例1

目标效果：在浏览器地址栏输入：www.123.com（这个是虚无的网址），跳转到linux系统的tomcat主页面中。

![](./figure/反向代理配置实例1的原理图.png)

### 修改本地映射

`C:/Windows/System32/drivers/etc/hosts`修改域名映射即可。

```txt
124.223.224.180 www.123.com
```

如果用的是云服务器，有可能这样采取的映射是无法达到目标效果的。云服务器让你备案：`www.123.com`

### nginx请求转发

```bash
# 修改 /usr/local/nginx/conf/nginx.conf 文件
server {
        listen       80;
        server_name  124.223.224.184;#nginx服务器地址
        location / {
            root   html;
            proxy_pass  http://127.0.0.1:8080; # tomcat服务器地址
            index  index.html index.htm;
        }

}

```

## 4.3 反向代理：实例2

目标效果：

使用nginx反向代理，根据访问路径跳转到不同端口的服务中

1. nginx监听端口：9001
2. 访问：9001/pinduoduo/ 直接映射到tomcat服务器127.0.0.1:8080/pinduoduo/index.html
3. 访问：9001/taobao/ 直接映射到tomcat服务器127.0.0.1:8081/taobao/index.html

### 准备两个tomcat服务器

```bash
# 杀死已经开启的tomcat服务器
ps -ef | grep tomcat

root     3711231       1  0 3月28 ?       00:00:58 /usr/bin/java -Djava.util.logging.config.file=/usr/local/apache-tomcat-8.5.77/conf/logging.properties -Djava.util.logging.manager=org.apache.juli.ClassLoaderLogManager -Djdk.tls.ephemeralDHKeySize=2048 -Djava.protocol.handler.pkgs=org.apache.catalina.webresources -Dorg.apache.catalina.security.SecurityListener.UMASK=0027 -Dignore.endorsed.dirs= -classpath /usr/local/apache-tomcat-8.5.77/bin/bootstrap.jar:/usr/local/apache-tomcat-8.5.77/bin/tomcat-juli.jar -Dcatalina.base=/usr/local/apache-tomcat-8.5.77 -Dcatalina.home=/usr/local/apache-tomcat-8.5.77 -Djava.io.tmpdir=/usr/local/apache-tomcat-8.5.77/temp org.apache.catalina.startup.Bootstrap start
root     4118774 4027353  0 22:42 pts/0    00:00:00 grep --color=auto tomcat

kill -9 3711231

ps -ef | grep tomcat

root     4120493 4027353  0 22:43 pts/0    00:00:00 grep --color=auto tomcat
# 说明已关闭

# 然后复制两个tomcat安装包的解压包就行
# cd /usr/local
# 将当前tomcat解压后的文件夹修改为tomcat8080
mv apache-tomcat-8.5.77 tomcat8080
# 然后再将tomcat8080文件夹复制一份，并命名为tomcat8081
cp -rf tomcat8080 tomcat8081

# 开启8080服务器，因为服务器默认就是8080
cd tomcat8080/bin
./startup.sh

# 开启8081服务器，这个需要修改tomcat8081的配置文件
cd ../../tomcat8081/conf
vim server.xml
# 这个配置文件里有以下几个地方需要修改端口，目的是为了不和刚刚启动的8080服务器冲突
# 分别修改为8015,8019,8081

# <Server port="8005" shutdown="SHUTDOWN">
# 8005端口是用来关闭TOMCAT服务的端口。
# <Connector port="8009" protocol="AJP/1.3" redirectPort="8443" />
# 连接器监听8009端口，负责和其他的HTTP服务器建立连接。在把Tomcat与其他HTTP服务器集成时，就需要用到这个连接器。
# <Connector port="8080" protocol="HTTP/1.1" connectionTimeout="20000" redirectPort="8443" />
# 连接器监听8080端口，负责建立HTTP连接。在通过浏览器访问Tomcat服务器的Web应用时，使用的就是这个连接器

cd ../../bin
./startup.sh

# 然后这两个服务器都都开启了，在浏览器中看一下，如果两个都能打开，那么服务就开好了。

# 在tomcat8080文件夹下的webapps下新建pinduoduo/index.html
# 在tomcat8081文件夹下的webapps下新建taobao/index.html
# 页面内容如下
```

```html
<!DOCTYPE html>
<html>
        <head>
              <meta charset="utf-8">
              <title>8081</title>
        </head>
        <body>
              <h1>这里是8081服务器</h1>
              <h2>淘宝</h2>
              <iframe src="https://www.taobao.com" width="800px" height="400px"></iframe>
        </body>
</html>
```



### 配置

`/usr/local/nginx/conf/nginx.conf`

```bash
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
        server {
                listen       80;
                server_name  124.223.224.184;
                location / {
                        root   html;
                        proxy_pass      http://127.0.0.1:8080;
                        index  index.html index.htm;
                }
        }
        server {
                listen  9001;
                server_name 124.223.224.184;
                location ~ /pinduoduo/ {
                        proxy_pass      http://127.0.0.1:8080;
                }
                location ~ /taobao/ {
                        proxy_pass      http://127.0.0.1:8081;
                }
        }
}
```

ps：记得对外开放9001，而无需开放8080和8081

```bash
cd ../sbin/
# 关闭nginx服务
./nginx -s stop
# 开启服务
./nginx
ps -ef | grep nginx
root      974231       1  0 20:33 ?        00:00:00 nginx: master process nginx
nobody    974232  974231  0 20:33 ?        00:00:00 nginx: worker process
root      974559  863111  0 20:33 pts/0    00:00:00 grep --color=auto nginx
```

然后浏览器访问：

`http://124.223.224.184:9001/taobao/index.html`

`http://124.223.224.184:9001/pinduoduo/index.html`

## 4.4 负载均衡

nginx负载均衡是由代理模块和上游（upstream）模块共同实现的。

### 实例

实现效果：

多个浏览器访问同一个nginx服务器地址，这多个请求被平均分配到不同的服务器上。

准备工作：

在两台tomcat服务器的webapps目录中，创建名称是webhome文件夹，再创建一个index.html，用于测试。

```bash
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
	upstream myserver {
         server  127.0.0.1:8080;
         server  127.0.0.1:8081;
    }
    server{
          listen  9002;
          server_name 124.223.224.184;
          location / {
                proxy_pass http://myserver;
          }
    }
}
```

浏览器输入：`http://124.223.224.184:9002/webHome/index.html`,

页面加载完成后，又刷新，他会在8080和8081服务器上来回跳跃。

### 负载均衡策略

1. 轮询（默认）

   - 每个请求按时间顺序逐一分配到不同的后端服务器。如果有服务器宕机，那么就剔除掉。

   - 加权轮询，平滑轮询。

   - ```bash
     upstream myserver {
     	server  127.0.0.1:8080 weight=1;#加权轮询
     	server  127.0.0.1:8081 weight=2;
     }
     ```

2. IP哈希

   - 每个请求按访问ip的hash结果分配，这样每个访客固定访问一个后端服务器，可以解决session的问题。

   - ```bash
     upstream myserver {
     	ip_hash;
     	server  127.0.0.1:8080;
     	server  127.0.0.1:8081;
     }
     ```

3. fair

   - 按后端服务器相应时间来分配请求，响应快的优先分配。

   - ```bash
     upstream myserver {
     	fair;
     	server  127.0.0.1:8080;
     	server  127.0.0.1:8081;
     }
     ```

4. 一致性哈希、最少连接、随机负载

## 4.5 动静分离

nginx动静分离简单来说就是把动态请求跟静态请求分开。使请求访问更加高效。

在一定程度上可以理解为，nginx处理静态页面，tomcat处理动态页面。不能理解成，只是单纯的把动态页面和静态页面物理分离。

动静分离从目前实现角度来讲大致分为两种：

1. 服务器分离：一种是纯粹把静态文件独立成单独的域名，放在独立的服务器上，也是目前主流推崇的方案。
2. 代理服务器分离：一种是动态文件和静态文件混合在一起发布，通过nginx分开。通过location指定不同的后缀名实现不同的请求转发。

### expire

通过expire参数设置，可以使浏览器缓存过期时间，减少与服务器之间的请求和流量。

具体Expire的定义：是给一个资源设定一个过期时间，也就是无需服务器去验证，直接通过浏览器自身确认是否过期即可，所以不会产生额外的流量，此种方法非常适合不经常变动的资源。

假设设置为3D，表示3d之内访问这个url，发送一个请求，比对服务器该文件的最后更新时间，如果更新时间未改变，则返回304（采用浏览器缓存）。如果有修改，则直接从服务器上重新下载，返回200。(协商缓存)

### 配置

在服务器下，建立一个静态文件夹，路径是`/usr/local/staticFileDirectory/`，在里面建一个www和images文件夹，www里放index.html，images里放index.jpg

```bash
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
    server {
                    listen       80;
                    server_name  124.223.224.184;
                    location /www/ {
                            root   /usr/local/staticFileDirectory/;
                            index  index.html index.htm;
                    }
                    location /images/ {
                            root /usr/local/staticFileDirectory/;
                            autoindex on;
                    }
     }
}
            
```

autoindex的作用是当你访问`http://124.223.224.184/images/`你可以看到`/usr/local/staticFileDirectory/images/`文件夹下的文件目录，如下。

![](./figure/静态文件autoindex.png)

如果访问不到图片或者html，还要看看是否是文件权限是否设置好了。

# 理论

# 1 核心配置指令

## 1.1 进程的核心配置指令

nginx的**进程**核心配置指令包含在Nginx核心代码及事件模块代码中。

按配置指令设定的功能可以分为**进程管理、进程调优、进程调试、事件处理**4个部分

```bash
# 1 进程管理
# nginx本身就是一款应用软件，在其运行时，用户可对其运行方式，动态加载模块，日志输出等使用其内建的基础配置指令进行配置。
daemon on;  # 以守护进程（服务）的方式运行nginx
pid logs/nginx.pid;	# 指定记录主进程的pid的文件路径
user nobody nobody;	# 工作进程运行的用户是
load_module "modules/ngx_http_xslt_filter_module.so"	# 加载动态模块
include - # 加载外部配置文件
error_log logs/error.log debug;	# 指定错误日志文件及文件名
pcre_jit on;	#用于设置配置文件中的正则表达式是否使用pcre_jit技术

# 2 进程调优
# nginx工作进程的性能依赖于硬件和操作系统的配置
# 在实际的应用场景中，用户需要按照硬件的、操作系统或应用场景需求的侧重点进行相应的配置调整。
thread_pool default thread=32 max_queue=65536;	# 线程池中的线程数为32，所有线程繁忙时，等待队列中的最大任务数为65536
timer_resolution 100ms;	# 处理事件超时的时间间隔
worker_priority -5;	# 设定进程在linux系统中的优先级（nice值）
worker_processes auto;	# 设置工作进程数，auto根据cpu内核数生成对应的工作进程数
worker_cpu_affinity auto;	# 将工作进程与cpu内核进行绑定
worker_rlimit_nofile 65535;	# nginx所有进程同时打开文件的最大数量，默认为操作系统的文件打开数
worker_shutdown_timeout 10s;	# 设置nginx正常关闭工作进程的超时时间，超过该时间是，强制关闭所有连接，以便关闭工作进程
lock_file logs/niginx.lock;	# 互斥锁文件，在开启accept_mutex进程调度模式或使用共享内存文件时，需要使用到互斥锁文件。

# 3 进程调试
master_process on;	# nginx默认是以主进程管理多个工作进程的工作方式，设定为off即运行一个主进程来处理所有请求，当只有一个主进程处理请求是，调试进程会更加方便。
working_directory logs;	# 该指令用于设定工作进程保存崩溃文件的目录，
worker_rlmit_core 800m;	# 设置崩溃文件的大小
debug_points stop;	# 该指令用于进行调试点的控制，当为stop时，在执行到调试点时或发出SIGSTOP信号，方便用户进行调试，当指令值为abort时，在调试点停止进程并创建corefile

# 4 事件处理
# nginx是采用事件驱动式架构处理外部请求的，这一架构使得nginx在现有硬件架构下可以处理数以万计的并发请求。
events {
	worker_connections 65535;	# 每个工作进程可处理的最大请求数。
	use epoll;	# 事件处理的机制模型
	accept_mutex on;	# 是否启用互斥锁模式的进程调度
	accept_mutex_delay 300ms;	# 在互斥锁模式下，工作进程需要不断地争抢互斥锁，抢锁等待时间设置为一个较短的时间，以提高争抢互斥的频率
	multi_accept on;	# 默认情况下每个工作进程一次只能接受一个新连接，如开启，则接收所有的新连接
	worker_aio_requests 128;	# 用于在epoll事件模型下使用aio时，单个工作进程未完成异步i/o操作的最大数
	debug_connection 192.0.2.0/24;	# 对指定客户端开启调试日志，在编译时需要开启--with-debug
}
```

