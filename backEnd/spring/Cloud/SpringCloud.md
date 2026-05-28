# SpringCloud

![image-20260520090518015](legend/image-20260520090518015.png)

参考文档：https://github.com/mofan212/spring-cloud-demo/blob/master/README.md

参考视频：https://www.bilibili.com/video/BV1UJc2ezEFU

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

## 1.2 服务注册

1. 在pom.xml中，引入 `spring-boot-starter-web`、`spring-cloud-starter-alibaba-nacos-discovery` 依赖，**记得添加后，要刷新引入，否则web服务无法启动**

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



## 1.3 服务发现

1. 开启服务发现，在主启动类上添加 `@EnableDiscoveryClient` 注解

2. 测试两款 API 的服务发现功能：`DiscoveryClient` 和 `NacosServiceDiscovery`。前者为 Spring 提供的服务发现标准接口，后者由 Nacos 提供。

   - 写一个测试类测试一下

   - 在pom.xml中引入test依赖

     ```xml
     <dependency>
         <groupId>org.springframework.boot</groupId>
         <artifactId>spring-boot-starter-test</artifactId>
         <scope>test</scope>
     </dependency>
     ```

   - `cloud-demo1/services/service-product/src/test/java/com.yanfang.product.DiscoveryTest.java`

     ```java
     package com.yanfang.product;
     
     import com.alibaba.cloud.nacos.discovery.NacosServiceDiscovery;
     import com.alibaba.nacos.api.exception.NacosException;
     import org.junit.jupiter.api.Test;
     import org.springframework.beans.factory.annotation.Autowired;
     import org.springframework.boot.test.context.SpringBootTest;
     import org.springframework.cloud.client.ServiceInstance;
     import org.springframework.cloud.client.discovery.DiscoveryClient;
     
     import java.util.List;
     
     
     @SpringBootTest
     public class DiscoveryTest {
         @Autowired
         DiscoveryClient discoveryClient;
     
         @Autowired
         NacosServiceDiscovery nacosServiceDiscovery;
     
         @Test
         void nacosServiceDiscoveryTest() throws NacosException{
             for(String service: nacosServiceDiscovery.getServices()){
                 System.out.println("service=" + service);
                 List<ServiceInstance> instances = nacosServiceDiscovery.getInstances(service);
                 for(ServiceInstance instance : instances){
                     System.out.println("server ip: "+instance.getHost() + "; port: "+instance.getPort());
     
                 }
             }
         }
     
         @Test
         void discoveryClientTest(){
             for(String service: discoveryClient.getServices()){
                 System.out.println("service=" + service);
                 List<ServiceInstance> instances = discoveryClient.getInstances(service);
                 for(ServiceInstance instance : instances){
                     System.out.println("server ip: "+instance.getHost() + "; port: "+instance.getPort());
     
                 }
             }
         }
     }
     
     
     /*打印出
     service=service-order
     server ip: 10.114.13.182; port: 8001
     server ip: 10.114.13.182; port: 8000
     service=service-product
     server ip: 10.114.13.182; port: 9002
     server ip: 10.114.13.182; port: 9001
     server ip: 10.114.13.182; port: 9000
     */
     ```

## 1.4 远程调用

![远程调用基本流程](legend/远程调用基本流程.svg)

<img src="legend/image-20260528103347711.png" alt="image-20260528103347711" style="zoom:67%;" />

## 1.5 负载均衡

引入依赖：

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-loadbalancer</artifactId>
</dependency>
```



### 1.5.1 使用 `LoadBalancerClient` 实现

```java
private Product getProductFromRemoteWithLoadBalancerClient(Long productId){
    //1、获取到商品服务所在的所有机器IP+port
    ServiceInstance choose = loadBalancerClient.choose("service-product");
    //远程URL
    String url = "http://"+choose.getHost() +":" +choose.getPort() +"/product/"+productId;
    log.info("远程请求：{}", url);
    //2、给远程发送请求
    Product product = restTemplate.getForObject(url, Product.class);
    return product;
}
```

### 1.5.2 使用 `@LoadBalanced` 注解实现

在配置类中向 Spring 容器添加 `RestTemplate` 的 Bean，在 Bean 方法上添加 `@LoadBalanced` 注解，使用 `RestTemplate` 进行远程调用时，修改传入的 URL 为服务名

```java
package com.yanfang.order.config;

import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;


@Configuration
public class OrderServiceConfig {

    @LoadBalanced
    @Bean
    public RestTemplate restTemplateBalancer(){
        return new RestTemplate();
    }
}
```

```java
private Product getProductFromRemoteWithLoadBalancerAnnotation(Long productId){
    //远程URL
    String url = "http://service-product/product/"+productId;
    //2、给远程发送请求：service-product会被动态替换
    Product product = restTemplateBalancer.getForObject(url, Product.class);
    //restTemplate那里需要用@LoadBalanced进行注解
    return product;
}
```

```java
// 在product中加HttpServletRequest request，可以查看请求地址信息

@GetMapping("/product/{id}")
public Product getProduct(@PathVariable("id") Long productId, HttpServletRequest request){
    String requestURL = request.getRequestURL().toString();
    System.out.println("请求完整地址："+requestURL);
    Product product = productService.getProductById(productId);
    return product;
}
```

### 1.5.3 面试题：如果注册中心宕机

经典面试题：如果注册中心宕机，远程调用是否可以成功？

- 如果从未调用过，此时注册中心宕机，调用会立即失败
- 如果调用过：
  - 此时注册中心宕机，会因为存在缓存的服务信息，调用会成功
  - 如果注册中心和对方服务都宕机，因为会缓存名单，调用会阻塞后失败（Connection Refused）

![](legend/远程调用步骤.svg)

## 1.6 配置中心

在大型分布式系统里面，配置中心可以用来管理所有微服务的配置。

想要修改配置，只需要在配置中心修改，就可以实时把变更的配置推送给指定的微服务，从而实现不停机配置更新。

1. 引入配置中心依赖

   ```xml
   <!--  配置中心，在services中的pom.xml   -->
   <dependency>
       <groupId>com.alibaba.cloud</groupId>
       <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
   </dependency>
   ```

2. 在各自的微服务中的`application.properties`中

   ```properties
   spring.application.name=service-order
   server.port=8000
   
   spring.cloud.nacos.server-addr=127.0.0.1:8848
   
   # 添加这一行
   spring.config.import=nacos:service-order.properties

3. 在nacos中，访问：`localhost:8848/nacos`，配置管理 -> 配置列表，创建配置

   ![image-20260528152055061](legend/image-20260528152055061.png)

### 1.6.1 @Value + @RefreshScope

**适用于少量配置**

1. 在微服务中访问配置

   ```java
   package com.yanfang.order.controller;
   
   import org.springframework.beans.factory.annotation.Autowired;
   import org.springframework.beans.factory.annotation.Value;
   import org.springframework.web.bind.annotation.GetMapping;
   import org.springframework.cloud.context.config.annotation.RefreshScope;
   import org.springframework.web.bind.annotation.RestController;
   
   @RefreshScope	// 当配置修改后，接口里面能获取到更新后的值
   @RestController
   public class OrderController {
   
   
       @Value("${order.timeout}")
       String orderTimeout;
       @Value("${order.auto-confirm}")
       String orderAutoConfirm;
   
       @GetMapping("/config")
       public String config(){
           return "order.timeout="+orderTimeout+";  order.auto-confirm="+orderAutoConfirm;
       }
   }
   ```

   

2. 重启微服务order

   ```bash
   # 会看见
   [Nacos Config] Load config[dataId=service-order.properties, group=DEFAULT_GROUP] success
   ...
   [Nacos Config] Listening config: dataId=service-order.properties, group=DEFAULT_GROUP
   ```

3. 浏览器访问：`http://localhost:8000/config`，返回：`order.timeout=30min; order.auto-confirm=7d`

4. 如果你在services中添加了配置中心的服务，在order微服务中使用了配置中心，而在product中没使用，那么就会在product重启的时候报下面的错误，你要在product的application.properties中，添加`spring.cloud.nacos.config.import-check.enabled=false`就不会再报错
   ```bash
   15:34:28.577 [main] ERROR org.springframework.boot.diagnostics.LoggingFailureAnalysisReporter -- 
   
   ***************************
   APPLICATION FAILED TO START
   ***************************
   
   Description:
   
   No spring.config.import property has been defined
   
   Action:
   
   Add a spring.config.import=nacos: property to your configuration.
   	If configuration is not required add spring.config.import=optional:nacos: instead.
   	To disable this check, set spring.cloud.nacos.config.import-check.enabled=false.
   ```

### 1.6.2 `@ConfigurationProperties` 无感自动刷新

1. 在`src/main/java/com/yanfang/order/properties/OrderProperties.java`下

   ```java
   package com.yanfang.order.properties;
   
   import lombok.Data;
   import org.springframework.boot.context.properties.ConfigurationProperties;
   import org.springframework.stereotype.Component;
   
   @Component
   @ConfigurationProperties(prefix = "order")      // 配置批量绑定，在nacos中，无需@RefreshScope就能实现自动刷新
   @Data
   public class OrderProperties {
       String timeout;
   
       String autoConfirm;     // 在properties中的短横线可以映射为驼峰写法
   }
   
   ```

   

2. 访问配置

   ```java
   @RestController
   public class OrderController {
   
   
       @Autowired
       OrderProperties orderProperties;
   
       @GetMapping("/config")
       public String config(){
           return "order.timeout="+orderProperties.getTimeout()+";  order.auto-confirm="+orderProperties.getAutoConfirm();
       }
   }
   ```



### 1.6.3 订阅配置的变化

场景：

1. 项目启动就监听配置文件变化
2. 发生变化后拿到变化的值
3. 发送邮件



```java
package com.yanfang.order;

import com.alibaba.cloud.nacos.NacosConfigManager;
import com.alibaba.nacos.api.config.ConfigService;
import com.alibaba.nacos.api.config.listener.Listener;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Bean;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

@EnableDiscoveryClient // 开启服务发现功能
@SpringBootApplication
public class OrderMainApplication {
    public static void main(String[] args){
        SpringApplication.run(OrderMainApplication.class, args);
    }

    @Bean
    ApplicationRunner applicationRunner(NacosConfigManager nacosConfigManager) {
        return args -> {
            ConfigService configService = nacosConfigManager.getConfigService();

            // 这里的dataId可以不是本微服务的properties，也可以是其他properties，只要配置中心有
            configService.addListener("service-order.properties", "DEFAULT_GROUP",
                    new Listener() {
                        @Override
                        public Executor getExecutor() {
                            return Executors.newFixedThreadPool(4);
                        }

                        @Override
                        public void receiveConfigInfo(String configInfo) {
                            System.out.println("变化的配置信息："+ configInfo);
                            System.out.println("邮件通知...");
                        }
                    }
            );
        };
    }
}

```

### 1.6.4 思考：Nacos与application.properties有相同的配置项，哪个生效

配置中心先生效

![](legend/配置信息优先级.svg)

```properties
spring.application.name=service-order
server.port=8000

spring.cloud.nacos.server-addr=127.0.0.1:8848

# 先导入的优先：service-order.properties > common.properties
spring.config.import=nacos:service-order.properties,nacos:common.properties
```



## 1.7 数据隔离

场景：

- 项目通常部署在多套环境上，比如 dev、test、prod。
- 每个微服务，同一种配置，在每套环境上的值可能不一样，eg：database.properties，common.properties
- 要求项目可以通过切换环境，加载本环境的配置。

如果要完成以上需求，其中的难点是如何：

- 区分多套环境
- 区分多种微服务
- 区分多种配置
- 按需加载配置

![](./legend/Nacos数据隔离解决方案.svg)

Nacos 的解决方案：

- 用NameSpace区分多套环境
- 用 Group 区分多种微服务
- 用 Data-id 区分多种配置
- 使用 SpringBoot 激活对应环境的配置



### 1.7.1 命名空间

访问：http://localhost:8848/nacos/，命名空间，新建命名空间

![image-20260528164456749](legend/image-20260528164456749.png)

**使用NameSpace区分多套环境**

![image-20260528164639450](legend/image-20260528164639450.png)

**使用group区分多种微服务**

![image-20260528165014460](legend/image-20260528165014460.png)

![image-20260528170239924](legend/image-20260528170239924.png)

### 1.7.2 激活环境

```yaml
server:
  port: 8000
spring:
  profiles:
# 这里激活
    active: test
  application:
    name: service-order
  cloud:
    nacos:
      server-addr: 127.0.0.1:8848
      config:
        import-check:
          enabled: false
        namespace: ${spring.profiles.active:public}

---
spring:
  config:
    import:
      - nacos:common.properties?group=order
      - nacos:database.properties?group=order
    activate:
      on-profile: dev

---
spring:
  config:
    import:
      - nacos:common.properties?group=order
      - nacos:database.properties?group=order
    activate:
      on-profile: test


---
spring:
  config:
    import:
      - nacos:common.properties?group=order
      - nacos:database.properties?group=order
    activate:
      on-profile: prod


```



# 2 OpenFeign

