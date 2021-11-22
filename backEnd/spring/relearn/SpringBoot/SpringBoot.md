# SpringBoot

# A 环境搭建

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

在springMVC中，Controller负责处理有DispatcherServlet接收并分发过来的请求，它把用户请求的数据通过业务处理层封装成一个Model，然后再把该Model返回给对应的View展示

### 3.1 控制器

#### 3.1.1 常用注解

| 注解            | 使用位置 | 说明                                                         |
| --------------- | -------- | ------------------------------------------------------------ |
| @Controller     | 类       | 表示是一个Controller对象，servlet分发处理器将会扫描使用该注解的类，并检测其中的方法是否使用了@RequestMapping方法，由它真正处理请求。 |
| @RestController | 类       | = @Controller + @ResponseBody                                |
| @RequestMapping | 类or方法 | 用来处理请求地址映射的注解，可以用在类上或方法上，如果用在类上则表示所有响应请求的方法都以该地址作为父路径。 |
| @PathVariable   | 参数     | 将请求URL中的模板变量映射到功能处理方法的参数上，            |

1. RequestMapping有6个属性
   - value：指定请求的地址
   - method：指定处理请求的方法
     - http请求的方法：GET，POST，PUT，DELETE，PATCH，OPTIONS，TRACE
   - consumer：消费消息，指定处理请求的提交内容类型（Content-Type），例如：application/json
     - HTTP中媒体的类型Content-Type，
     - 常见媒体格式：
       - text/xml：XML格式，还有html等
       - text/plain：纯文本
       - image/png：png图片格式，jpg，gig等
     - 以application开头的媒体格式
       - application/json
       - application/xhtml+xml
       - application/pdf，application/msword（word文档格式）
       - application/octet-stream：二进制流数据（常用于文件下载）
       - multipart/form-data：如果在表单中进行文件上传，则需要使用该格式
       - application/x-www-form-urlencoded，表单数据编码方式，`<form encType=" " >`中默认的encType，Form(表单)被默认编码为key/value格式给服务器
   - produces：生产消息，指定返回的内容类型。仅当request请求头中的Accept类型中包含该指定类型时才返回。
   - params：指定request中必须包含某些参数值才让该方法处理请求
   - headers：指定request中必须包含某些指定的header值才让该方法处理请求
2. 常用restful风格请求映射的注解
   - @GetMapping：处理GET请求，相当于`@RequestMapping(value = "", method = RequestMethod.GET )`
   - @PostMapping
   - @DeleteMapping
   - @PutMapping
3. 

#### 3.1.2 在方法中使用参数

```java
@GetMapping("/article/{id}")
public String getArticleContent(@PathVariable("id") Integer id){
    //获取http://localhost:8080/article/10021中，10021
}
@RequestMapping("/addUser")
public String addUser(String username){
    //获取http://localhost:8080/user/addUser?username=qqq中，username变量的值qqq
}
@RequestMapping("/addUser")
public String addUser(UserModel user){
    //直接通过model将获得的数据映射为model对象
}
@RequestMapping( value = "/addUser", method=RequestMethod.POST)
public String addUser(@ModelAttribute("user") UserModel user){
    //用于从Model、Form或Url请求参数中获取对应属性的对应值
}
@RequestMapping("/addUser")
public String addUser(HttpServletRequest request){
    System.out.println("name:" + request.GETParameter("username"));
    //通过HttpServletRequest接收参数
}
@RequestMapping("/addUser")
public String addUser(@RequestParam(value="username",required=false) String name){
    //通过注解RequestParam绑定参数，
    //一个请求，只有一个RequestBody；一个请求，可以有多个RequestParam。
}
@RequestMapping("/addUser",method=RequestMethod.POST)
@ResponseBody
public String saveUser(@RequestBody List<UserModel> users){
    //   @RequestBody主要用来接收前端传递给后端的json字符串中的数据的(请求体中的数据)
}
//还有获取图片，获取文件等映射，这里不再叙述，后面有具体需要了再看

```

### 3.2 模型

模型Model在MVC模式中是实体Bean。

其作用暂时存储数据于内存中，以便进行持久化，以及在数据变化时更新控制器。

简单来说，Model是数据库表对应的实体类。

#### 3.2.1 验证数据

Hibernate-validator可实现数据的验证，它是对JSR标准的实现。

在web开发中，不需要额外为验证在导入此依赖，web依赖已集成了Hibernate-validator。

web依赖集成了如下一些依赖

- spring-boot-starter
- spring-boot-starter-json
- spring-boot-starter-tomcat
- hibernate-validator
- spring-web
- spring-webmvc

validator验证的常用注解

| 注解                       | 作用类型           | 说明                                          |
| -------------------------- | ------------------ | --------------------------------------------- |
| @NotBlank( message=  )     | 字符串             | 非null，且length>0                            |
| @Email                     | 字符串             | 被注解的元素必须符合电子邮箱的格式            |
| @Length(min= ,max= )       | 字符串             | 字符串长度控制                                |
| @NotEmpty                  | 字符串             | 字符串必须非空                                |
| @NotEmptyPattern           | 字符串             | 非空并且匹配正则表达式                        |
| @DateValidator             | 字符串             | 验证日期格式是否满足正则表达式，local为英语   |
| @DateFormatCheckPattern    | 字符串             | 验证日期格式是否满足正则表达式，local自行指定 |
| @CreditCardNumber          | 字符串             | 验证信用卡号码                                |
| @Range(min=,max=,message=) | 字符串，数值，字节 | 验证范围                                      |
| @Null                      | 任意               | 必须为null                                    |
| @NotNull                   | 任意               | 不为null                                      |
| @AssertTrue                | 布尔值             | 必须为True                                    |
| @AssertFalse               | 布尔值             | 为false                                       |
| @Min(value)                | 数字               | 为数字，并且不小于                            |
| @Max(value)                | 数字               | 为数字，并且不大于                            |
| @DecimalMin(value)         | 数字               | 为数字，并且不小于                            |
| @DecimalMax(value)         | 数字               | 为数字，并且不大于                            |
| @Size(max=,min=)           | 数字               | 为数字，在指定范围                            |
| @Digits(integer,fraction)  | 数字               |                                               |
| @Past                      | 日期               | 过去的日期                                    |
| @Future                    | 日期               | 未来的日期                                    |
| @Pattern(regex=,flag=)     | 正则表达式         | 必须符合正则表达式                            |
| @LastStringPattern         | `List<String>`     | 验证集合中字串是否符合正则表达式              |

##### 自定义验证

自定义验证需要提供两个类：

1. 自定义注解类
2. 自定义验证业务逻辑实现类

```java
//1.自定义验证注解类
import com.example.MyCustomConstraintValidator;
import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

//限定使用范围
@Target({ElementType.FIELD})
//表明注解的生命周期，它在代码运行时可以通过反射获取到注解
@Retention(RetentionPlicy.RUNTIME)
//@Contraint注解，里面传入了一个validatedBy字段，以指定该注解的校验逻辑
@Contraint(validatedBy = MyCustomConstraintValidator.class)//自定义验证业务逻辑实现类
public @interface MyCustomConstraint{
	String message() default "请输入中国政治或经济中心的城市名";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
        
}

//2.自定义验证业务逻辑实现类
import com.example.demo.MyCustomConstraint;
import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;
public class MyCustomConstraintValidator implements ConstraintValidator<MyCustomConstraint, String>{//String为校验的类型
    @Override
    public void initialize(MyCustomConstraint myConstraint){
        //在启动时执行
    }
    //自定义校验逻辑
    @Override
    public boolean isValid(String s, ConstraintValidatorContext validatorContext){
        if(!(s.equals("北京")|| s.equals("上海"))){
            return false;
        }
        return true
    }
}
```

#### 3.2.2 创建实体

```java
import com.example.demo.MyCustomConstrant;
import lombok.Data;
import org.hibernate.validator.constraints.Length;
import javax.validation.constraints.*;
import java.io.Serializable;
@Data
public class User implements Serializable{
    private Long id;
    
    @NotBlank(message="用户名不能为空")
    @Length(min=5, max=20,message="用户名长度为5-20个字符")
    private String name;
    
    @NotNull(message="年龄不能为空")
    @Min(value=18,message="最小18岁")
    @Max(value=80,message="最大60岁")
    private Integer age;
    
    @Email(message="请输入邮箱")
    @NotBlank(message="邮箱不能为空")
    private String email;

    @MyCustomContraint
    private String answer;
}
```

#### 3.2.3 实现控制器

## 4 响应式编程

以餐厅"叫号"来比喻阻塞式编程与响应式编程

阻塞式编程：假设一个餐厅没有叫号机或者前台服务员，以餐台的数量来安排客人。店里有200个餐台，并且此时已坐满，那么最后一个客人就直接被拒绝服务了。

响应式编程：店里有200个餐台，并且此时已坐满，后面再来客人，叫号机马上给后面的每个客人一个排队号，这样服务就不会堵塞，每个人立马能得到反馈。来再多的人也能立马给排号，但用餐依然是阻塞的。

程序上：假设服务器最大线程资源数为200个，当前遇到200个非常耗时的请求，如果再来一个请求，阻塞式程序就已经无法处理（拒绝服务）。而响应式程序，则可以立即响应（告诉用户等着），然后将收到的请求转发给work线程去处理，主要应用场景：在业务处理较耗时的场景中，减少服务器资源的占用，提高并发处理速度。

结论：MVC能满足的场景，就不需要用WebFlux，如果开发I/O密集型服务，则可以选择用WebFlux实现。

### 4.1 Mono和Flux

Mono和Flux是Reactor中的两个概念：

- Mono和Flux属于事件发布者，为消费者（前端请求）提供订阅接口，当有事件发生时，Mono或Flux会回调消费者相应方法（onComplete()-排队结束，onNext()-排队中，onError()-排队出错）
- Mono和Flux用于处理异步数据流

# C 进阶篇

## 5 springBoot进阶

### 5.1 AOP

AOP把业务功能分为

- 核心业务：增删改数据库，用户登录
- 非核心业务：性能统计，日志，事务管理

将非核心业务功能被定义为切面，然后将切面和核心业务功能编织在一起，这就是切面

在面向对象编程的过程中，我们很容易通过继承、多态来解决纵向扩展。 但是对于横向的功能，比如，在所有的service方法中开启事务，或者统一记录日志等功能，面向对象的是无法解决的。所以AOP——面向切面编程其实是面向对象编程思想的一个补充。

AOP的核心概念

1. 切入点（pointcut）：在哪些类、哪些方法上切入
2. 通知（advice）：在方法前、方法后、方法前后做什么
3. 切面（Aspect）：切面 = 切入点 + 通知
4. 织入（weaving）：把切面加入对象，并创建出代理对象的过程
5. 环绕通知

```java
@Aspect //使之成为切面类
@Component //把切面类交由ioc容器管理
public class AopLog{
    private Logger logger = LoggerFactory.getLogger(this.getClass());
	//线程局部变量，用于解决多线程中相同变量的访问冲突问题
    ThreadLocal<Long> startTime = new ThreadLocal<>();
    
    //定义切点，用于决定在什么时候执行此切入点，我猜测当执行某个controller的时候，就会主动执行这个切入点
    @PointCut("execution( public.... )") //具体配置需要查阅资料
    public void aopWebLog(){
    	//核心业务
    }
    
    //在切入点开始前切入的内容
    @Before("aopWebLog()")
    public void doBefore(JoinPoint joinPoint) throws Throwable{
        startTime.set(System.currentTimeMillis());
        
        //接收到请求，记录请求内容
        ServletRequestAttribute attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        HttpServletRequest request = attributes.getRequest();
        
        //记录下请求的内容
        logger.info("URL：" + request.getRequestURL().toString());
        logger.info("HTTP 方法：" + request.getMethod());
        logger.info("IP地址：" + request.getRemoteAddr());
        logger.info("类的方法：" + joinPoint.getSignature().getDeclaringTypeName() + "." + joinPoint.getSignature().getName());
        logger.info("参数：" + request.getQueryString());
    }
    
    //在切入点返回内容之后切入内容，可以对返回结果做一些处理
    @AfterReturning(pointcut = "aopWebLog()", returning = "retObject")
    public void doAfterReturning(Object retObject) throws Throwable{
        //处理完请求，返回内容
        logger.info("应答值：" + retObject);
        logger.info("费时：" + (System.currentTimeMills() - startTime.get()));
    }
    
    //在切入点抛出异常之后，做一些处理
    @AfterThrowing(pointcut = "aopWebLog()", throwing = "ex")
    public void addAfterThrowingLogger(JoinPoint joinPoint, Expection ex){
    	logger.error("执行 " + "异常" ,ex)
    }
    
    // @Around：在切入点前后切入内容
    // @After：在切入点末尾切入内容
}
```

### 5.2 IOC

Inversion of Control容器，是面向对象的一种设计原则，意为控制反转，他将程序创建对象的控制权交给spring框架来管理，以便降低代码耦合度。

控制反转的实质是获得依赖对象的过程被反转了，这个过程有自身管理变为由IoC容器主动注入，这正是IoC实现的方式之一：依赖注入（DI，Dependency Injection）

![](./legend/ioc容器的优点.png)

IOC的实现方法主要有两种：

1. 依赖注入：
   - IOC容器通过**类型或名称**等信息将不同对象注入不同属性中。依赖注入主要有以下几种方式：
     - 设置注入（setter injection）：让IOC容器调用注入所依赖类型的对象
     
     - 接口注入（interface injection）：实现特定的接口，以供Ioc容器注入所依赖类型的对象
     
     - 构造注入（constructor injection）：实现特定参数的构造函数，在创建对象时让IOC容器注入所依赖的对象
     
     - 基于注解：通过@Autowired等注解让IOC容器注入所依赖的对象，@Controller，@Service，@Component
     
     - 以下为spring的做法，现在springboot里面基本都只会使用注解注入
     
     - ```xml
       <?xml version="1.0" encoding="UTF-8"?>
       <beans xmlns="http://www.springframework.org/schema/beans"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
           
           
           <bean id="math" class="com.test2.MathDemo"/>
           <bean id="english" class="com.test2.EnglishDemo"/>
           
           <!--属性注入MathDemo和StudyDemo-->
           <bean id="StudyDemo" class="com.test2.StudyDemo">
               <property name="mathDemo" ref="math"/>
               <property name="englishDemo" ref="english"/>
           </bean>
       
           <!--构造注入MathDemo和StudyDemo-->
           <bean id="StudyDemo" class="com.test2.StudyDemo">
               <constructor-arg ref="math"/>
               <constructor-arg ref="english"/>
           </bean>
        
       </beans>
       ```
     
     - ```java
       package com.test2;
        
       public class MathDemo {
           public void StudyMath(){
               System.out.println("学习数学。。。。");
           }
       }
       public class EnglishDemo {
           public void StudyMath(){
               System.out.println("学习英语。。。。");
           }
       }
       
       
       public class StudyDemo {
           private MathDemo mathDemo;
           private EnglishDemo englishDemo;
           public void study(){
               mathDemo.StudyMath();
               englishDemo.StudyEnglish();
           }
           
           public StudyDemo(){
        
           }
           //构造注入里面必须有对应的构造方法
           public StudyDemo(MathDemo mathDemo,EnglishDemo englishDemo){
               this.mathDemo=mathDemo;
               this.englishDemo=englishDemo;
           }
           
           //setter注入
           public void setMathDemo(MathDemo mathDemo) {
               this.mathDemo = mathDemo;
           }
           public void setEnglishDemo(EnglishDemo englishDemo) {
               this.englishDemo = englishDemo;
           }
       
       }
       ```
     
     - 
   
2. 依赖查找：

   - 这个后期完善

![](./legend/三种注入方式的比较.png)

#### [ioc基于注解注入的例子](https://www.cnblogs.com/ye-hcj/p/9613491.html)

```java
//语言的接口
public interface Language{
    public String getGreeting();
    public String getBye();
}
//语言实现类
public class Chinese implements Language{
    @Override
    public String getGreeting(){
        return "你好";
    }

    @Override
    public String getBye() {
        return "再见";
    }
}
public class English implements Language{
    @Override
    public String getGreeting(){
        return "Hello";
    }
    @Override
    public String getBye() {
        return "Bye bye";
    }
}

// 此类就是ioc容器中的一个bean，内部属性通过外部注入
// @Service的作用就是声明他是一个bean
// @Autowired的作用就是依赖注入
package com.springlearn.learn.bean;
@Service
public class GreetingService{

    @Autowired
    private Language language;

    public GreetingService() {

    }

    public void sayGreeting() {
        String greeting = language.getGreeting();
        System.out.println("Greeting:" + greeting);
    }
}

//AppConfiguration
// 此类是一个定义bean和集中bean的文件
// @Configuration声明这个类是定义bean的
// @ComponentScan扫描bean目录
// @Bean(name="language") 定义了一个名为language的bean，只要访问此bean就会自动调用getLanguage方法
package com.springlearn.learn.config;
@Configuration
@ComponentScan({"com.springlearn.learn.bean"})
public class AppConfiguration{

    @Bean(name="language")
    public Language getLanguage() {
        return new Chinese();
    }
}


@Repository
public class MyRepository{
    public String getAppName(){
        return "Hello my first Spring App";
    }

    public Date getSystemDateTime() {
        return new Date();
    }
}

@Component
public class MyComponent {
    @Autowired
    private MyRepository repository;

    public void showAppInfo(){
        System.out.println("Now is:" + repository.getSystemDateTime());
        System.out.println("App Name" + repository.getAppName());
    }
}

//DemoApplication
public class DemoApplication {
	public static void main(String[] args) {
		ApplicationContext context = new 				 	       AnnotationConfigApplicationContext(AppConfiguration.class);

		System.out.println("-------------");

		Language language = (Language)context.getBean("language");
		System.out.println("Bean Language: "+ language);
        	//Bean Language: com.springlearn.learn.langimpl.Chinese@258d79be
		System.out.println("Call language.sayBye(): "+ language.getBye());
        	//Call language.sayBye(): 再见

		GreetingService service = (GreetingService) context.getBean("greetingService");
        //获取bean时，getBean首字母小写是spirng规定的（byName）。可以参阅spring的2.7.1小节
        //为什么这里获取的就是中文bean?
		service.sayGreeting();//Greeting:你好
		System.out.println("----------");
        
        MyComponent myComponent = (MyComponent) context.getBean("myComponent");
		myComponent.showAppInfo();
        //Now is:Sun Sep 09 13:48:45 CST 2018
		//App NameHello my first Spring App
    }
}
```

### 5.3 [Servlet](https://blog.csdn.net/fg881218/article/details/89716366)

**Servlet 是 javax.servlet 包中定义的接口。**

它声明了 Servlet 生命周期的三个基本方法：init()、service() 和 destroy()。它们由每个 Servlet Class（在 SDK 中定义或自定义）实现，并由服务器在特定时机调用。

- **init()** 方法在 Servlet 生命周期的初始化阶段调用。它被传递一个实现 javax.servlet.ServletConfig 接口的对象，该接口允许 Servlet 从 Web 应用程序访问初始化参数。
- **service()** 方法在初始化后对每个请求进行调用。每个请求都在自己的独立线程中提供服务。Web容器为每个请求调用 Servlet 的 service() 方法。service() 方法确认请求的类型，并将其分派给适当的方法来处理该请求。
- **destroy()** 方法在销毁 Servlet 对象时调用，用来释放所持有的资源。

从 Servlet 对象的生命周期中，我们可以看到 **Servlet 类是由类加载器动态加载到容器中的**。每个请求都在自己的线程中，Servlet 对象可以同时服务多个线程（线程不安全的）。当它不再被使用时，会被 JVM 垃圾收集。
像任何Java程序一样，Servlet 在 JVM 中运行。为了处理复杂的 HTTP 请求，Servlet 容器出现了。Servlet 容器负责 Servlet 的创建、执行和销毁。

#### Servlet 容器和 Web 服务器如何处理一个请求的

1. Web 服务器接收 HTTP 请求。
2. Web 服务器将请求转发到 Servlet 容器。
3. 如果对应的 Servlet 不在容器中，那么将被动态检索并加载到容器的地址空间中。
4. 容器调用 init() 方法进行初始化（仅在第一次加载 Servlet 时调用一次）。
5. 容器调用 Servlet 的 service() 方法来处理 HTTP 请求，即读取请求中的数据并构建响应。Servlet 将暂时保留在容器的地址空间中，可以继续处理其它 HTTP 请求。
6. Web 服务器将动态生成的结果返回到浏览器/客户端。

![](./legend/servlet请求响应机制.png)

#### 自定义Servlet类

在使用springboot 应用程序时，使用Controller基本能解决大部分的功能需求，但有时也需要使用Servlet，比如实现拦截和监听功能。

springboot的核心控制器DispatcherServlet会处理所有的请求。如果自定义Servlet，则需要进行注册，以便DispatcherServlet核心控制器知道它的作用，以及处理请求url-pattern

使用Servlet处理请求，可以直接通过注解@WebServlet(urlPattern, description)注册Servlet，然后在入口类中添加注解@ServletComponentScan，以扫描该注解指定包下的所有Servlet

```java
//注册Servlet类
@WebService(urlPattern = "/servletDemo/*")//属性urlPattern指定WebServlet的作用范围
public class ServletDemo02 extends HttpServlet{
    //父类的doGet方法是空的，子类需要重写此方法
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException{
        System.out.println("doGet");
        resp.getWriter().print("Servlet servletDemo");
    }
}

//开启Servlet支持
@ServletComponentScan
@SpringBootApplication
public class ServletDemoApplication{
	public static void main(String[] args){
        SpringApplication.run(ServletDemoApplication.class,args);
    }
}
```

### 5.4 [三大器](https://www.cnblogs.com/hhhshct/p/8808115.html)

过滤器（Filter）、监听器（Listener）、拦截器（Interceptor ）都属于面向切面编程的具体实现。

#### 5.4.1 Filter

Filter是依赖于Servlet容器，属于Servlet规范的一部分

过滤器：在很多Web应用中，都会用到过滤器，如参数过滤，防止sql注入、防止页面攻击、空参数矫正、Token验证、Session验证、点击率统计等。

Filter的使用步骤

1. 实现Filter抽象类
2. 重写Filter抽象类里的init、doFilter、destory方法
3. 通过在入口类添加注解@ServletComponentScan，以注册Filter类

```java
//如果有多个Filter，则序号越小，越早被执行
@order(1)
//url过滤配置
@WebFilter(filterName="Filter01", urlPattern="/*")
public class Filter01 implements Filter{
    @Override
    public void init(FilterConfig filterConfig) throw ServletException{
        //init逻辑，该init将在服务器启动时调用
    }
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throw IOException, ServletException{
        //请求request 处理逻辑
        //请求request 封装逻辑
        //chain 重新写回request和response
        filterChain.doFilter(servletRequest,servletResponse);
    }
    @Override
    public void destroy(){
        //重写destroy逻辑，该逻辑将在服务器关闭时被调用
    }
}
```



#### 5.4.2 Listener

要可用于以下方面：1、统计在线人数和在线用户2、系统启动时加载初始化信息3、统计网站访问量4、记录用户访问路径。

Servlet中的监听器分为以下3中类型：

1. 监听ServletContext、Request、Session作用域的创建和销毁
   - ServletContextListener，
   - HttpSessionListener，监听新的Session创建事件
   - ServletRequestListener，监听ServletRequest的初始化和销毁
2. 监听ServletContext、Request、Session作用域中属性的变化（增删改）
   - ServletContextAttributeListener：监听Servlet上下文参数变化
   - HttpSessionAttributeListener：监听HttpSession参数变化
   - ServletRequestAttributeListener：监听ServletRequest参数的变化
3. 监听HttpSession中对象状态的改变（被绑定、解除绑定、钝化、活化）
   - HttpSessionBindingListener：监听HttpSession，并绑定及解除绑定
   - HttpSessionActivationListener：监听钝化和活动的HttpSession状态改变

通过实现以上监听类的监听方法，然后在入口类添加注解@ServletComponentScan

#### 5.4.3 Interceptor

过滤器与拦截器的区别：

1. Filter是依赖于Servlet容器，属于Servlet规范的一部分，而拦截器则是独立存在的，可以在任何情况下使用。
2. Filter的执行由Servlet容器回调完成，而拦截器通常通过动态代理的方式来执行。
3. Filter的生命周期由Servlet容器管理，而拦截器则可以通过IoC容器来管理，因此可以通过注入等方式来获取其他Bean的实例，因此使用会更方便。



拦截器：Interceptor 在AOP中用于在某个方法或字段被访问之前，进行拦截然后在之前或之后加入某些操作。比如日志，安全等。**一般拦截器方法都是通过动态代理的方式实现。**可以通过它来进行日志记录、权限检查，或者判断用户是否登陆，或者是像12306 判断当前时间是否是购票时间。

### 5.5 自动配置

### 5.6 元注解

元注解就是定义注解的注解。

元注解就是在自定义注解的时候，给自定义注解使用的注解。

| 元注解      | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| @Retention  | 声明是注解类，该注解用于说明自定义注解的生命周期             |
| @Target     | 生命该注解能够注解的目标                                     |
| @Inherited  | 该注解是一个标记注解，表明被该元注解注解的自定义注解在修饰其他类的时候，这个自定义注解同样可以作用于这个其他类的子类上 |
| @Documented | 该注解是一个标记注解，表明被该元注解注解的自定义注解会被javadoc工具记录。 |
| @Interface  | 定义注解，定义体内的每个方法实际上都声明了一个配置参数，方法名就是参数名称，返回值的类型就是参数的类型，并且可以通过default来声明参数的默认值。 |

1. [@Retention](https://blog.csdn.net/m0_37840000/article/details/80921775)

   - 使用示例：@Retention(RetentionPolicy.RUNTIME)
   - 生命周期的值是一个枚举值，
     - RetentionPolicy.RUNTIME，始终存在，注解不仅保存到class文件中，而且jvm加载class之后，仍然存在
     - RetentionPolicy.CLASS，注解被保留到class文件中，但在jvm加载class文件时候被遗弃，这就是默认的生命周期
     - RetentionPolicy.SOURCE，注解只保留在源文件中，当java文件被编译成class文件的时候，注解被遗弃。所以他们不会被写入字节码中。@Override、@SuppressWarnings都属于这类注解。
   - 首先要明确生命周期长度 **SOURCE < CLASS < RUNTIME** ，所以前者能作用的地方后者一定也能作用。一般如果需要**在运行时去动态获取注解信息，那只能用 RUNTIME 注解**；如果要**在编译时进行一些预处理操作**，比如生成一些辅助代码（如 ButterKnife）**，就用 CLASS注解**；如果**只是做一些检查性的操作**，比如 **@Override** 和 **@SuppressWarnings**，则**可选用 SOURCE 注解**。

2. [@Target](https://blog.csdn.net/fengcai0123/article/details/90544338)

   - 使用示例：@Target(ElementType.FIELD)
   - 作用目标的类型是一个枚举值
     - ElementType.CONSTRUCTOR，可以用于修饰构造器
     - ElementType.FIELD，可以用于修饰成员变量，对象，属性
     - ElementType.LOCAL_VARIABLE
     - ElementType.METHOD
     - ElementType.PACKAGE
     - ElementType.PARAMETER
     - ElementType.TYPE，可以修饰class、interface、enum
   - 声明自定义注解的修饰类型后，如果被修饰对象不符，将会报错。

3. [@Interface](https://blog.csdn.net/zhangbeizhen18/article/details/87885441/)

   - ```java
     @Retention(RetentionPolicy.RUNTIME)  
     public @interface MyAnnotation  
     {  
      String hello() default "gege";  
       String world();  
       int[] array() default { 2, 4, 5, 6 };  
       EnumTest.TrafficLamp lamp() ;  
       TestAnnotation lannotation() default @TestAnnotation(value = "ddd");  
       Class style() default String.class;  
     }
     /*
     上面程序中，定义一个注解@MyAnnotation，定义了6个属性，他们的名字为：  
     
     hello,world,array,lamp,lannotation,style.  
     
     属性hello类型为String,默认值为gege  
     属性world类型为String,没有默认值  
     属性array类型为数组,默认值为2，4，5，6  
     属性lamp类型为一个枚举,没有默认值  
     属性lannotation类型为注解,默认值为@TestAnnotation，注解里的属性是注解  
     属性style类型为Class,默认值为String类型的Class类型
     */
     
     /*
     如果注解中有一个属性名字叫value,则在应用时可以省略属性名字不写。  
     可见，@Retention(RetentionPolicy.RUNTIME )注解中，RetentionPolicy.RUNTIME是注解属性值，属性名字是value
     */  
     
     public @interface MyTarget  
     {  
         String value();  
     }
     //这里value参数的值就是aaa
     @MyTarget("aaa")  
      public void doSomething()  
      {  
       System.out.println("hello world");  
      }  
     ```

   - 

   ```java
   //自定义一个可以修饰属性的注解
   @Documented
   @Retention(RetentionPolicy.RUNTIME)
   @Target(ElementType.FIELD)
   public @interface FiledAnnotation {
   	 String value() default "GetFiledAnnotation";   
   }
   //自定义一个可以修饰方法的注解
   @Documented
   @Retention(RetentionPolicy.RUNTIME)
   @Target(ElementType.METHOD)
   public @interface MethodAnnotation {
   	String name() default "MethodAnnotation";   
       String url() default "https://www.cnblogs.com";
   }
   //自定义一个可以修饰class、interface、enum的注解
   @Documented
   @Retention(RetentionPolicy.RUNTIME)
   @Target(ElementType.TYPE)
   public @interface TypeAnnotation {
   	String value() default "Is-TypeAnnotation";
   }
   
   //使用上述的自定义注解
   @TypeAnnotation(value = "doWork")
   public class Worker {
    
   	@FiledAnnotation(value = "CSDN博客")
   	private String myfield = "";
    
   	@MethodAnnotation()
   	public String getDefaultInfo() {
   		return "do the getDefaultInfo method";
   	}
    
   	@MethodAnnotation(name = "百度", url = "www.baidu.com")
   	public String getDefineInfo() {
   		return "do the getDefineInfo method";
   	}
   }
   
   //可以通过泛型获得注解上的值
   public class TestMain {
   	
   	public static void main(String[] args) throws Exception {
   		
           Class cls = Class.forName("com.zbz.annotation.pattern3.Worker");
           Method[] method = cls.getMethods();
           /**判断Worker类上是否有TypeAnnotation注解*/
           boolean flag = cls.isAnnotationPresent(TypeAnnotation.class);
           /**获取Worker类上是TypeAnnotation注解值*/
           if (flag) {
           	TypeAnnotation typeAnno = (TypeAnnotation) cls.getAnnotation(TypeAnnotation.class);
           	System.out.println("@TypeAnnotation值:" + typeAnno.value());
           }
           
           /**方法上注解*/
           List<Method> list = new ArrayList<Method>();
           for (int i = 0; i < method.length; i++) {
               list.add(method[i]);
           }
           
           for (Method m : list) {
           	MethodAnnotation methodAnno = m.getAnnotation(MethodAnnotation.class);
               if (methodAnno == null)
                   continue;
               System.out.println( "方法名称:" + m.getName());
               System.out.println("方法上注解name = " + methodAnno.name());
               System.out.println("方法上注解url = " + methodAnno.url());
           }
           /**属性上注解*/
           List<Field> fieldList = new ArrayList<Field>();
           for (Field f : cls.getDeclaredFields()) {// 访问所有字段
           	FiledAnnotation filedAno = f.getAnnotation(FiledAnnotation.class);
           	System.out.println( "属性名称:" + f.getName());
           	System.out.println("属性注解值FiledAnnotation = " + filedAno.value());
           } 
   	}
    
   }
   ```


### 5.7 异常处理

异常分类：

1. Error：代表编译和系统错误，不允许捕获
2. Exception：标准java库的方法引发的异常
3. Runtime Exception：运行时异常
4. Non_RuntimeException：非运行时异常
5. Throw：用户自定义异常

异常处理方式：

1. 捕获异常：`try{}catch(e){}`
2. 抛出异常：
   - throw：
     - 如果需要在程序中自行抛出异常，则应使用throw语句
     - throw抛出的不是异常类，而是一个异常实例，并且每次只能抛出一个异常实例
   - throws：
     - throws声明抛出异常只能在方法签名中使用，可以抛出多个异常，多个异常类间用逗号隔开
     - 如果某段代码中调用了一个带throws声明的方法，该方法声明抛出Checked异常，则表明，该方法希望它的调用者来处理该异常。也就是说，调用该方法时，要么放在try块中显示捕获该异常，要么放在另一个带throws声明抛出的方法中
3. 自定义异常

#### 控制器通知

Spring提供一个非常方便的异常处理方案——控制器通知（@ControllerAdvice 或 RestControllerAdvice），它将所有控制器作为一个切面，利用切面技术实现。

通过基于@ControllerAdvice或@RestControllerAdvice 的注解可以对异常进行全局统一处理，默认对所有Controller生效。也可以通过注解的参数来配置**生效范围**。

- 按注解：`@ControllerAdvice(annotations = RestController.class)`
- 按包名：`@ControllerAdvice("org.example.controller")`
- 按类型：`@ControllerAdvice(assignableTypes = {ControllerInterface.class})`

通过@ControllerAdvice注解可以实现多个异常处理类

异常处理类会包含一个或多个被下面的注解注释的方法，这些注解不是异常处理类独有。

- @InitBinder，对表单数据进行绑定，用于定义控制器参数绑定规则
- @ModelAttribute：在控制器方法被执行前，对所有Controller的Model添加属性进行操作
- @ExceptionHandler：定义控制器发生异常后的操作，可以拦截所有控制器发生的异常
- @ControllerAdvice：统一异常处理，通过@ExceptionHandler(value = Exception.class)来指定捕获的异常。@ControllerAdvice + @ExceptionHandle 可以处理除404以外的运行异常。

```java
//自定义异常类
public class BusinessException extends RuntimeException{
    //自定义错误码
    private Integer code;
    //自定义构造器，必须输入错误码及内容
    public BusinessException(int code,String msg) {
        super(msg);
        this.code = code;
    }
    public Integer getCode() {
        return code;
    }
    public void setCode(Integer code) {
        this.code = code;
    }
}
//异常处理类
@ControllerAdvice
public class CustomerBusinessExceptionHandler {
    @ResponseBody //抛错后，异常处理类直接返回相应的data
    @ExceptionHandler(BusinessException.class)
    public Map<String, Object> businessExceptionHandler(BusinessException e) {
        Map<String, Object> map = new HashMap<String, Object>();
        map.put("code", e.getCode());
        map.put("message", e.getMessage());
        //发生异常进行日志记录，此处省略
        return map;
    }
}

//测试
@RestController
public class TestController {
    @RequestMapping("/BusinessException")
    public String testResponseStatusExceptionResolver(@RequestParam("i") int i){
        if (i==0){
            throw new BusinessException(600,"自定义业务错误");
        }
             return "success";
    }
}
```

## 6  ORM操作数据库

对象关系映射ORM（Object Relational Mapping）。

### 6.1 JDBCTemplate

JDBC（Java DataBase Connectivity），它是java用于连接数据库的规范，实际上，它由一组用Java语言编写的类和接口组成，为大部分关系型数据库提供访问接口。

JDBC需要每次进行数据库连接，然后处理SQL语句、传值、关闭数据库，这里过程多，也容易出错，为了减少工作量，减少失误，JDBCTemplate就被设计出来了。

JDBCTemplate是对JDBC的封装，它更便于程序的实现，再不需要每次都进行连接、打开、关闭了。

配置基础依赖

```xml
<!--jdbcTemplate依赖-->
<dependency>
	<groupId>org.springframework.boot</groupId>
    <artfactId>spring-boot-start-jdbc</artfactId>
</dependency>
<!--MySql数据库依赖-->
<dependency>
	<groupId>mysql</groupId>
    <artfactId>mysql-connector-java</artfactId>
    <scope>runtime</scope>
</dependency>
```

```properties
//配置数据库
spring.datasource.url=jdbc:mysql://127.0.0.1........
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
```



```java
//新建Entity实体类
@Data
public class User implements RowMapper<User> {
    private int id;
    private String username;
    private String password;
    @Override
    public User mapRow(ResultSet resultSet, int i) throws SQLException{
        User user = new User();
        user.setId(resultSet.getInt("id"));
        user.setUsername(resultSet.getString("username"));
        user.setPassword(resultSet.getString("password"));
        return user;
    }
}
/*
*/
```

JDBCTemplate提供了以下3种操作数据的方法

- execute：用于执行sql语句，用于DDL。
- update：包括新增、修改、删除操作。用于DML的增删改。
- query：查询，用于DML的查。

```java
@SpringBootTest
@RunWith(SpringRunner.class)
public class UserControllerTest {
    @Autowired
    private JdbcTemplate jdbcTemplate;
	// 创建表,execute
    @Test
    public void createUserTable() throws Exception {
        String sql = "CREATE TABLE `user` (\n" +
                "  `id` int(10) NOT NULL AUTO_INCREMENT,\n" +
                "  `username` varchar(100) DEFAULT NULL,\n" +
                "  `password` varchar(100) DEFAULT NULL,\n" +
                "  PRIMARY KEY (`id`)\n" +
                ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;\n" +
                "\n";
        jdbcTemplate.execute(sql);
    }
    // 添加数据
    @Test
    public void saveUserTest() throws Exception  {
        String sql = "INSERT INTO user (USERNAME,PASSWORD) VALUES ('longzhiran','123456')";
        int rows = jdbcTemplate.update(sql);
        System.out.println(rows);
    }
    // 查数据
     @Test
    public void list() throws Exception{
        String sql = "SELECT * FROM user_jdbct";
        List<User> userList = jdbcTemplate.query(sql,
                new BeanPropertyRowMapper(User.class));
        for (User userLists : userList) {
            System.out.println(userLists);
        }
    }
}
```

JDBCTemplate学习成本低，会sql就能上手，虽操作麻烦但很容易学会。

JDBCTemplate实现起来比ORM繁琐，所以大部分开发者用的都是ORM。

### 6.2 ORM

对象关系映射ORM（Object Relational Mapping），将数据库中的表和内存中的对象建立映射关系。对象和关系型数据库是业务实体的两种表现形式，业务实体在内存中表现为对象，在数据库中表现为关系型数据，内存中的对象不会被永久保存，只有关系型数据库的对象会被永久保存。

ORM系统一般以中间键的形式存在，因为内存中的对象存在关联和继承关系，而数据库中，关系型数据无法直接表达多对多的关联和继承关系。

![](./legend/ORM映射.png)

