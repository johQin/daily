# SpringBoot

# 1 环境搭建

## 1.1 安装和配置

1. jdk1.8 
   - 安装，配置环境变量，java -version
2. maven
   - 安装，配置环境变量，校验是否安装完成mvn -v
   - 配置国内仓库
3. idea
   - 配置maven环境`file->setting，搜索maven，`
     - `maven home directory（安装地址），`
     - `user settings file（自定义的maven配置文件，里面配置有国内镜像仓库的配置）`
     - `local repository(存放通过maven管理的依赖包，的目录地址)`

## 1.2 helloworld

1. 创建项目：`new Project->maven(next)->填写项目的具体信息（name，groupId，ArtifactId，Version，项目存放位置location，Finish）`

2. 项目依赖：

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>
   
       <groupId>com.sifangtech</groupId>
       <artifactId>helloworld01</artifactId>
       <version>1.0.0</version>
   
       <parent>
           <groupId>org.springframework.boot</groupId>
           <artifactId>spring-boot-starter-parent</artifactId>
           <version>2.3.4.RELEASE</version>
       </parent>
       <properties>
           <maven.compiler.source>8</maven.compiler.source>
           <maven.compiler.target>8</maven.compiler.target>
       </properties>
       <dependencies>
           <dependency>
               <groupId>org.springframework.boot</groupId>
               <artifactId>spring-boot-starter-web</artifactId>
           </dependency>
       </dependencies>
   <!--打jar包要用的插件-->
       <build>
           <plugins>
               <plugin>
                   <groupId>org.springframework.boot</groupId>
                   <artifactId>spring-boot-maven-plugin</artifactId>
               </plugin>
           </plugins>
       </build>
   </project>
   ```

   

3. 编写应用：

   - main.java.com.sifangtech.boot.mainApplication.java

   - ```java
     package com.sifangtech.boot;
     
     import org.springframework.boot.SpringApplication;
     import org.springframework.boot.autoconfigure.SpringBootApplication;
     
     /**
      * 注解告诉springboot这是一个springboot应用
      * */
     @SpringBootApplication
     public class mainApplication {
         public static void main(String[] args) {
             SpringApplication.run(mainApplication.class,args);
         }
     }
     ```

   - main.java.com.sifangtech.boot.controller.HelloController

   - ```java
     package com.sifangtech.boot.controller;
     
     import org.springframework.web.bind.annotation.RequestMapping;
     import org.springframework.web.bind.annotation.RestController;
     
     @RestController
     public class HelloController {
     
         @RequestMapping("/hello")
         public String sayHello() {
             return "hello, spring boot";
         }
     }
     
     ```

4. 应用配置文件

   - main.resources.application.properties

   - ```properties
     server.port=8082
     #具体相关配置可以参看：https://docs.spring.io/spring-boot/docs/2.2.5.RELEASE/reference/html/appendix-application-properties.html#server-properties
     ```

5. 运行：

   - 右击main.java.com.sifangtech.boot.mainApplication.java即可运行

6. 浏览器：`localhost:8082/hello`

7. 部署：

   - 在依赖文件中引入打包插件，打成jar，直接在目标服务器上，`java -jar`即可部署
   - 在cmd的命令行中，要取消掉快速编辑模式

8. 



## 1.坑

1. 项目的`pom.xml`中引入`dependency`时，无法引入，在idea中显示是红色。
   - 注意在idea中配置的maven环境可能因为新建工程项目后发生改变，好像可以在新建的项目中固定这个配置，但暂时不知道如何固定maven在idea中的配置。
   - 所以当无法引入某些依赖包时，需要去查看idea中关于maven的配置是否发生改变。
2. 无法引入插件，在idea中显示是红色，需要在右侧的maven工具栏点击刷新，这样可以重新导入依赖

# B 基础篇

## 2 springboot 基础

### 2.1 spirngboot项目框架结构

1. `src/main/java:`入口（启动）类及程序的开发目录。在这个目录下进行业务开发、创建实体层、控制器层、数据连接层（`Service、Entity、Controller、Mapper`）
   - `XxxApplication.java`：入口类，项目的启动入口
2. `src/main/resource`：资源文件目录，主要用于放静态文件和配置文件
   - `/static`：用于存放静态资源，css、js、img等
   - `/templates`：用于存放模板文件
   - `application.properties、application.yml`：用于配置项目运行所需的配置数据
3. `src/test/java`：放置测试程序
4. `target/`：项目打包文件
5. `pom.xml`：**Project Object Model 的缩写，即项目对象模型**，pom.xml 就是 maven 的配置文件，用以描述项目的各种信息。和依赖信息。[详情](https://zhuanlan.zhihu.com/p/76874769)

### 2.2 注解

注解用来定义一个类、属性或一些方法，以便程序能被编译处理。注解可以标注包、类、方法和变量等。

有系统注解、使用在类名上的注解、使用在方法上的注解、其他注解。注解众多，随着学习。

系统注解：`@Override、@Deprecated`

#### 2.2.1 常用在类名上的注解

| 注解            | 使用位置                     | 说明                                                         |
| --------------- | ---------------------------- | ------------------------------------------------------------ |
| @RestController | 类名上                       | = @ResponseBody + @Controller                                |
| @Controller     | 类名上                       | 声明此类为SpringMVC Controller对象                           |
| @Service        | 类名上                       | 声明一个业务处理类（实现非接口类）                           |
| @Repository     | 类名上                       | 声明数据库访问类                                             |
| @Component      | 类名上                       | 代表其是Spring管理类                                         |
| @Configuration  | 类名上                       | 声明此类是一个配置类                                         |
| @Resource       | 类名上、属性或构造函数参数上 | spring中按byName自动注入对象Bean                             |
| @AutoWired      | 类名上、属性或构造函数参数上 | spring中按byType自动注入对象Bean                             |
| @RequestMapping | 类名或方法上                 | 如果用在类上，表示所有响应请求的方法都是以该地址作为父路径的 |
| @Transactional  | 类名或方法上                 | 用于处理事务                                                 |
| @Qualifier      | 类名或属性上                 | 为Bean指定名称，随后在通过名字引用Bean                       |

#### 2.2.2 使用在方法上的注解

| 注解          | 使用位置   | 说明                                                         |
| ------------- | ---------- | ------------------------------------------------------------ |
| @RequestBody  | 方法参数前 | 将前端的请求参数转化为指定类型的实例                         |
| @PathVariable | 方法参数前 | 获取url路径中的参数                                          |
| @Bean         | 方法上     | 声明该方法返回的结果是一个由Spring容器管理的Bean             |
| @ResponseBody | 方法上     | 将控制器中方法返回的对象转化为指定格式（JSON/XML），写入Response对象的body数据区，给前端 |

#### 2.2.3 其他注解

| 注解                     | 使用位置      | 说明                                                         |
| ------------------------ | ------------- | ------------------------------------------------------------ |
| @EnableAutoConfiguration | 入口类/类名上 | 提供自动配置                                                 |
| @SpringBootApplication   | 入口类/类名上 | 用来启动入口类Application                                    |
| @Aspec                   | 入口类/类名上 | 标注切面。可以用来配置事务，日志、权限校验，在用户请求时做一些处理 |
| @EnableScheduling        | 入口类/类名上 | 用来开启计划任务，Spring通过@Scheduled支持多种类型的计划任务 |
| @ComponentScan           | 入口类/类名上 | 用来扫描组件，可自动发现和装配一些bean                       |
| @ControllerAdvice        | 类名上        | 包含@Component，可以被扫描到，统一处理异常                   |
| @ExpectionHandler        | 方法上        | 表示遇到这个异常就执行该方法                                 |
| @Value                   | 属性上        | 用于获取配置文件中的值                                       |
|                          |               |                                                              |

### 2.3 配置文件

SpringBoot支持使用Properties和Yaml两种格式文件的配置方式。但Properties的优先级高于Yaml格式文件。

| 注解                                   | 使用位置 | 说明                                                         |
| -------------------------------------- | -------- | ------------------------------------------------------------ |
| @SpringBootTest                        | 类       | 用于测试的注解，可指定入口类或测试环境等                     |
| @RunWith( SpringRunner.class )         | 类       | 在Spring测试环境中进行测试                                   |
| @Test                                  | 方法     | 表示一个测试方法                                             |
| @Value                                 | 属性上   | 用于获取配置文件中的值                                       |
| @ConfigurationProperties( prefix = "") | 类       | 把同前缀的配置信息自动封装为一个实体类                       |
| @Data                                  | 类       | 自动生成Setter、Getter、toString、equals、hashCode方法和无参构造器 |

#### 配置多环境

yml配置文件：

- 主配置文件application.yml、

- 开发环境application-dev.yml、

- 生产环境application-prod.yml

- ```yml
  #生产环境，application-prod.yml
  server:
  	port: 8080
  	servlet:
  		session:
  			timeout: 30
  	tomcat: 
  		uri-encoding:UTF-8
  myenvironment:
  	name: 生产环境
  
  #开发环境，application-dev.yml
  server:
  	port: 8080
  	servlet:
  		session:
  			timeout: 30
  	tomcat: 
  		uri-encoding:UTF-8
  myenvironment:
  	name: 开发环境
  
  #主配置文件，application.yml
  spring:
  	profiles:
  		active: dev #指定使用哪一个环境的配置文件
  ```

properties配置文件：

- 主配置文件application.properties、
- 开发环境application-dev.properties、
- 生产环境application-prod.properties
- 和yml同样

### 2.4 starter

springBoot为了简化配置，提供了非常多的Starter。在对应的`pom.xml`中配置即可。

常用starter

| starter                        | 说明 |
| ------------------------------ | ---- |
| spring-boot-starter-web        |      |
| spring-boot-starter-validation |      |
| spring-boot-starter-security   |      |
| spring-boot-starter-websocket  |      |
| spring-boot-starter-data-redis |      |
| spring-boot-starter-jdbc       |      |
| spring-boot-starter-mail       |      |
|                                |      |
|                                |      |
|                                |      |
|                                |      |
|                                |      |
|                                |      |
|                                |      |
|                                |      |

## 3 分层开发web应用

`SpringMVC(Model-View-Controller)`，

- Model：是java的实体Bean，
- View：主要用来解析、处理、显示渲染内容
- Controller：处理视图中的响应，决定如何调用Model的实体bean、如何调用Service层

![](./legend/springbootLevel.png)