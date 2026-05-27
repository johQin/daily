# SpringCloud

![image-20260520090518015](legend/image-20260520090518015.png)

# 0 绪论

## 0.1 架构演进

### **单体架构**

![image-20260520091526302](legend/image-20260520091526302.png)

### **集群架构**

它指的是一种物理形态

解决大并发问题。

缺点：模块化升级，多语言团队

![image-20260520091729295](legend/image-20260520091729295.png)

### **微服务架构**

一个大型应用被拆分成很多小应用，分布部署在各个机器

- 微服务应用：SpringBoot
- 注册中心/配置中心：Spring Cloud Alibaba Nacos
  - 服务发现和注册
  - 配置变更的主动下发
- 网关：Spring Cloud Gateway
- 远程调用（RPC）：Spring Cloud OpenFeign
  - 微服务之间的相互调用
- 服务熔断：Spring Cloud Alibaba Sentinel
  - 当一个服务请求另一个服务发生阻塞时（远程调用期间），为了防止这种阻塞导致的链式反应（服务雪崩），而执行的快速失败机制
  - eg：订单服务请求支付服务，前5秒支付服务只成功了50%，为了防止订单服务的阻塞，那么后5秒，直接让订单服务请求的支付服务直接失败
- 分布式事务：Spring Cloud Alibaba Seata
  - 服务拆分后，那么数据库也进行了拆分，不同数据库相关的数据如果存在事务关系，那么就需要用到分布式事务

![](./legend/微服务.png)

## 0.2 微服务

传统项目：**单体架构**

- 所有功能（用户、订单、支付、商品）全部写在一个项目里
- 打包一个 jar /war，部署一台服务器
- 缺点：牵一发动全身、改个小功能要全量发布、不好扩容

**微服务架构**：

把一个庞大的单体项目，**按业务拆分**成一个个独立、小型、自治的小服务：

- 用户服务、订单服务、商品服务、支付服务…
- 每个服务**独立开发、独立打包、独立部署、独立扩容**
- 服务之间通过 HTTP / RPC 互相调用协作



**一个微服务 = 一个 SpringBoot 项目**

- 用户微服务 → 一个 SpringBoot 工程
- 订单微服务 → 另一个 SpringBoot 工程
- 每个服务单独启动、单独运行、互不影响



完整技术栈层级（从上到下）

- 架构层：**微服务架构**（思想）
- 代码层：**SpringBoot**（写单个服务）
- 治理层：SpringCloud / SpringCloudAlibaba（服务注册、发现、熔断、网关、配置中心）



SpringCloud 是一套**微服务治理全家桶**，专门解决：**几十个 SpringBoot 微服务拆分开之后，互相怎么调用、怎么管控、怎么保证稳定**。

SpringCloud 是基于 SpringBoot 的**微服务治理框架合集**，提供服务注册发现、负载均衡、网关路由、熔断降级、配置中心、分布式事务、链路追踪等全套能力，用来解决微服务拆分后，服务通信、统一管控、高可用、分布式协作的问题。

## 0.3 [版本说明](https://github.com/alibaba/spring-cloud-alibaba/wiki/%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)

由于 Spring Boot 3.0，Spring Boot 2.7~2.4 和 2.4 以下版本之间变化较大，所以社区为此专门做了适配。

教程推荐版本：

- SpringBoot：3.3.4
- SpringCloud：2023.0.3
- SpringCloud Alibaba：2023.0.3.2

组件版本：

- Nacos：2.4.3
- Sentinel：1.8.8
- Seata：2.2.0

JDK17

Maven：3.9.16

项目结构：

![image-20260527172841315](legend/image-20260527172841315.png)

# 1 [Nacos](https://nacos.io/)

Nacos （Dynamic Naming and Configuration Service）一个更易于构建云原生应用的动态服务发现、配置管理和服务管理平台。

## 1.1 安装

安装：

- 下载最新的 Nacos 安装包，本文使用 Nacos-2.5.1
- 启动命令：`startup.cmd -m standalone`，`-m, --mode`

下载好的安装包后，解压到非中文目录，进入 `bin` 目录，执行启动命令。

```bash
startup.cmd -m standalone
"nacos is starting with standalone"

         ,--.
       ,--.'|
   ,--,:  : |                                           Nacos 2.4.3
,`--.'`|  ' :                       ,---.               Running in stand alone mode, All function modules
|   :  :  | |                      '   ,'\   .--.--.    Port: 8848
:   |   \ | :  ,--.--.     ,---.  /   /   | /  /    '   Pid: 25644
|   : '  '; | /       \   /     \.   ; ,. :|  :  /`./   Console: http://10.114.13.182:8848/nacos/index.html
'   ' ;.    ;.--.  .-. | /    / ''   | |: :|  :  ;_
|   | | \   | \__\/: . ..    ' / '   | .; : \  \    `.      https://nacos.io
'   : |  ; .' ," .--.; |'   ; :__|   :    |  `----.   \
|   | '`--'  /  /  ,.  |'   | '.'|\   \  /  /  /`--'  /
'   : |     ;  :   .'   \   :    : `----'  '--'.     /
;   |.'     |  ,     .-./\   \  /            `--'---'
'---'        `--`---'     `----'

2026-05-27 20:13:55,209 INFO Tomcat initialized with port(s): 8848 (http)
```

访问：http://localhost:8848/nacos，

- 配置管理：配置中心

## 1.2 注册

1. 在pom.xml中，引入 `spring-boot-starter-web`、`spring-cloud-starter-alibaba-nacos-discovery` 依赖

   - `spring-cloud-starter-alibaba-nacos-discovery`

     ```xml
     <!-- 在services.xml 中，nacos服务发现 -->
     <dependency>
         <groupId>com.alibaba.cloud</groupId>
         <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
     </dependency>
     ```

   -  `spring-boot-starter-web`

     ```xml
     <!-- 在service-order.xml 中，这是搭建web服务器所需要的依赖 -->
     <dependency>
         <groupId>org.springframework.boot</groupId>
         <artifactId>spring-boot-starter-web</artifactId>
     </dependency>
     ```

2. 编写web应用：

   `cloud-demo1/services/service-order/src/main/java/com/yanfang/order/OrderMainApplication.java`

   ```java
   package com.yanfang.order;
   
   import org.springframework.boot.SpringApplication;
   import org.springframework.boot.autoconfigure.SpringBootApplication;
   
   @SpringBootApplication
   public class OrderMainApplication {
       public static void main(String[] args){
           SpringApplication.run(OrderMainApplication.class, args);
       }
   }
   ```

3. 编写配置文件

   `cloud-demo1/services/service-order/src/main/resources/application.properties`

   ```properties
   spring.application.name=service-order
   server.port=8000
   
   spring.cloud.nacos.server-addr=127.0.0.1:8848
   ```

4. 运行OrderMainApplication

5. 访问 `http://localhost:8848/nacos/`，服务管理 -> 服务列表

   ![image-20260527204551881](legend/image-20260527204551881.png)

6. 开启多个微服务

   ![](./legend/开启多个微服务.png)

7. 查看nacos

   ![image-20260527212254251](legend/image-20260527212254251.png)

8. 