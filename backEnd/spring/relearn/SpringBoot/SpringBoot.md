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