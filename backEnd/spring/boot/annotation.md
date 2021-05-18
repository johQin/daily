# springBoot

框架层级：

1. Controller：控制捕获前台的请求，调用服务Service，控制返回相应请求。
2. Model：作为前台数据的映射，校验数据字段是否符合要求。
3. Service：用作复杂的后台逻辑，如需修改数据库调用相应的Mapper。
4. Entity：数据库实体。
5. Mapper：作为Model与Entity之间的映射（前台数据和数据库字段之间的映射关系）

# 1. spring org.springframework.web.bind.annotation

属于springMVC的内容

1. 处理request

   - @RequestBody

   - @RequestHeader
   - @RequestMethod
   - @RequestMapping
   - @RequestParam
   - @RequestPart
   - @CookieValue
   - @PathVariable;

2. 处理response
   - @ResponseBody
   - @ResponseStatus
3. 处理attribute属性
   - @SessionAttributes,
   - @ModelAttribute; 
4. 处理Exception
   - @ExceptionHandler
5. 实现aop
   - @ControllerAdvice
6. 其他处理
   - @InitBinder
   - @Mapping
   - @MatrixVariable
   - @Restcontroller
   - @ValueConstants

## 1.1 request注解

1. [@RequestBody](<https://blog.csdn.net/justry_deng/article/details/80972817>)

   - 主要用来接收前端传递给后端的json字符串中的数据的(请求体中的数据的)；
   - @RequestBody接收数据时，用POST方式进行提交。
   - @RequestBody与@RequestParam()可以同时使用，@RequestBody最多只能有一个，而@RequestParam()可以有多个。
   - RequestBody 接收的是请求体里面的数据；而RequestParam接收的是key-value里面的参数（postman中的选项params，也可以是url路径中的动态参数）
   - **@JsonAlias**与**@JsonProperty**指定模型中的属性对应什么key

2. [@RequestHeader](<https://blog.csdn.net/vsg123/article/details/78956757>)

   - @RequestHeader注解用于映射请求头数据到Controller方法的对应参数。
   - @RequestHeader注解与使用@RequestParam一样，在方法参数前加上注解即可

3. [@RequestMapping](<https://blog.csdn.net/zalan01408980/article/details/82904126>)

   - 用于映射请求路径，将请求URL映射到整个类上，某个特定的方法上
   - 一般来说，类级别的注解负责将一个特定（或符合某种模式）的请求路径映射到一个控制器上，同时通过方法级别的注解来细化映射，即根据特定的HTTP请求方法（GET、POST 方法等）、HTTP请求中是否携带特定参数等条件，将请求映射到匹配的方法上
   - 可以通过@PathVariable("") 注解将占位符中的值绑定到方法参数上
   -  一共有五种映射方式

4. [@RequestParam](<https://blog.csdn.net/sswqzx/article/details/84195043>)

   - 接收的是key-value里面的参数（postman中的选项params，也可以是url路径中的动态参数）

   - 将请求参数绑定到你控制器的方法参数上（是springmvc中接收普通参数的注解）
   - @RequestParam(value=”参数名”,required=”是否必传”,defaultValue=”默认值”)

5. [@RequestPart](<https://blog.csdn.net/wd2014610/article/details/79727061>)

   - @RequestPart这个注解用在multipart/form-data表单提交请求的方法上。
   - @RequestParam适用于name-valueString类型的请求域，@RequestPart适用于复杂的请求域（像JSON，XML）。

6. [@CookieValue](<https://www.cnblogs.com/east7/p/10303180.html>)

   - @CookieValue注解主要是将请求的Cookie数据，映射到功能处理方法的参数上。
   - @CookieValue(value=”参数名”,required=”是否必传”,defaultValue=”默认值”)

7. [@PathVariable](<https://blog.csdn.net/RAVEEE/article/details/89531299>)

   - @PathVariable 可以将 URL 中占位符参数绑定到控制器处理方法的入参中：URL 中的 {xxx} 占位符可以通过@PathVariable(“xxx“) 绑定到操作方法的入参中。

## 1.2 **Response注解**

1. [@ResponseBody](<https://blog.csdn.net/originations/article/details/89492884>)
   - @responseBody注解的作用是将controller的方法返回的对象通过适当的转换器转换为指定的格式之后，写入到response对象的body区，通常用来返回JSON数据或者是XML数据。
   - 在使用 @RequestMapping后，返回值通常解析为跳转路径，但是加上 @ResponseBody 后返回结果不会被解析为跳转路径
2. [@ResponseStatus](<https://blog.csdn.net/yalishadaa/article/details/71480694>)
   - 带有@ResponseStatus注解的异常类会被ResponseStatusExceptionResolver 解析。可以实现自定义的一些异常,同时在页面上进行显示。

# 2. javax.persistence

**Spring Data：**一个用于简化数据库访问，并支持云服务的开源框架，根据JPA规范封装的一套JPA应用框架（在底层。主要目标：是使得构建基于 Spring 框架应用对数据的访问变得方便快捷。

**JPA：**全称Java Persistence API，是sun提出的一个对象持久化规范。是一套对象关系映射规范(ORM：Object Relational Mapping)

JPA仅仅是一种规范，也就是说JPA仅仅定义了一些接口，而接口是需要实现才能工作的。所以底层需要某种实现，而Hibernate就是实现了JPA接口的ORM框架。springdata-jpa底层封装了Hibernate的jpa，理解为 JPA 规范的再次封装抽象。

JPA常用注解：

1. [@Entity]()

   - 标注用于实体类声明语句之前，指出该Java 类为实体类，将映射到指定的数据库表。

2. [@Table](<https://blog.csdn.net/sswqzx/article/details/84337672>)
   - @Table(**name** = "tab_user",**uniqueConstraints** = {@UniqueConstraint(columnNames={"uid","email"})})，、
     - **name** 用来命名当前实体类对应的数据库表的名字 
     - **uniqueConstraints** 用来批量命名唯一键，其作用等同于多个：@Column(unique = true)

3. [@Basic](<https://blog.csdn.net/weixin_37968613/article/details/100763966>)
   - 类型包含java基本类型（byte，short，int，long,float，double,char,boolean），包装类，枚举类，以及实现了Serializable接口的类型。
   - @Basic(**fetch**="属性的加载机制",**optional**="属性是否可null")
   - fetch有两个选项：EAGER（即时加载，默认值）和LAZY（懒加载），即时加载意味着当实例化对象的时候必须加载该属性值，懒加载是指当实例化对象时不加载该属性，只有当调用该属性时才加载。
   - optional用来指定属性是否可空,有两个选项：true（可空，默认值）和false
   - 如果你在实体类属性上不加@Basic注解，它也会自动加上@Basic，并使用默认值。

4. [@Column](<https://blog.csdn.net/qq_16769857/article/details/80347459>)

   - 用来标识实体类中属性与数据表中字段的对应关系
   - 属性特别多name，unique，nullable，length，insertable，updatable，table，precision和scale，columnDefinition
   - name，定义了被标注字段在数据库表中所对应字段的名称；
   - length，表示字段的长度，当字段的类型为varchar时，该属性才有效，默认为255个字符。
   - insertable和updatable属性一般多用于只读的属性，例如主键和外键等。这些字段的值通常是自动生成的。
   - precision属性和scale属性表示精度，当字段类型为double时，precision表示数值的总长度，scale表示小数点所占的位数。
     - columnDefinition表示创建表时，该字段创建的SQL语句，eg：@Column(name="del_flag",columnDefinition = "varchar(255) comment '删除标志 0:未删除 1:已删除' ")`

5. [@Id](<https://blog.csdn.net/coding1994/article/details/79597057>)

   - @Id 标注用于声明一个实体类的属性映射为数据库的主键列。该属性通常置于属性声明语句之前，可与声明语句同行，也可写在单独行上。 

6. [@GeneratedValue](<https://blog.csdn.net/u011781521/article/details/72210980>)

   - JPA主键通用策略生成器，

   - @GeneratedValue(strategy,generator)
   - generator  : String  //JPA 持续性提供程序为它选择的主键生成器分配一个名称
   - @GeneratedValue表示主键是自动生成策略，通过strategy 属性指定。一般该注释和 @Id 一起使用。默认情况下，JPA 自动选择一个最适合底层数据库的主键生成策略：SqlServer对应identity，MySQL 对应 auto increment。 
   -  在javax.persistence.GenerationType中定义了以下几种可供选择的策略： 
     - IDENTITY：采用数据库ID自增长的方式来自增主键字段，Oracle 不支持这种方式； 
     - AUTO： JPA自动选择合适的策略，是默认选项； 
     - SEQUENCE：通过序列产生主键，通过@SequenceGenerator 注解指定序列名，MySql不支持这种方式 
     - TABLE：通过表产生主键，框架借由表模拟序列产生主键，使用该策略可以使应用更易于数据库移植。

   - 

7. @GenericGenerator

   - @GenericGenerator注解----自定义主键生成策略
   - @GenericGenerator(name,strategy,parameters)。name属性指定生成器名称。 strategy属性指定具体生成器的类名。 parameters得到strategy指定的具体生成器所用到的参数。 
   - 一般配合@GeneratorValue使用

8. [@Transient](<https://blog.csdn.net/rongxiang111/article/details/86476028>)

   - @Transient表示该属性并非一个到数据库表的字段的映射，只能修饰属性
   - 变量将不再是对象持久化的一部分，该变量内容在序列化后无法获得访问。

9. [@Temporal](https://www.cnblogs.com/meng-ma-blogs/p/8474175.html)

   - Temporal注解的作用就是帮Java的Date类型进行格式化
   - 数据库的字段类型有date、time、datetime，
   - @Temporal(TemporalType.DATE)——>实体类会封装成日期“yyyy-MM-dd”的 Date类型。
   - TemporalType.TIME——>“hh-MM-ss”的 Date类型。
   - TemporalType.TIMESTAMP——>“yyyy-MM-dd hh:MM:ss”的 Date类型。

10. [@OneToMany](<https://blog.csdn.net/qq_38157516/article/details/80146547>)

    - 表的对应关系，一对多，通常使用jpa的接口操作，要看看jpa

11. **@ManyToOne**

12. **@ManyToMany**

13. **[@Query](<https://www.cnblogs.com/zhaobingqing/p/6864223.html>)** (Spring Data JPA 用法)

    - @Query注解查询适用于所查询的数据无法通过关键字查询得到结果的查询。这种查询可以摆脱像关键字查询那样的约束。这是Spring Data的特有实现。

    - ```
      @Query("SELECT p FROM Person p WHERE p.lastName = ?1 AND p.email = ?2")
      //只能写select语句
      ```

14. **@Modifying**

    - 配合@Query使用，使@Query里可以写Update和Deletesql语句，但是没有InsertSQL语句

**Java实体对象为什么一定要实现Serializable接口呢?**

**Serializable**

- 是java.io包中定义的、用于实现Java类的序列化操作而提供的一个语义级别的接口。Serializable序列化接口没有任何方法或者字段，只是用于标识可序列化的语义。

- 实现了Serializable接口的类可以被ObjectOutputStream转换为字节流，同时也可以通过ObjectInputStream再将其解析为对象。例如，我们可以将序列化对象写入文件后，再次从文件中读取它并反序列化成对象，也就是说，可以使用表示对象及其数据的类型信息和字节在内存中重新创建对象。

# 3 javax.validation.constraints

| 验证注解                                     | 验证的数据类型                                               | 说明                                                         |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| @AssertFalse                                 | Boolean,boolean                                              | 验证注解的元素值是false                                      |
| @AssertTrue                                  | Boolean,boolean                                              | 验证注解的元素值是true                                       |
| @NotNull                                     | 任意类型                                                     | 验证注解的元素值不是null                                     |
| @Null                                        | 任意类型                                                     | 验证注解的元素值是null                                       |
| @Min(value=值)                               | BigDecimal，BigInteger, byte,short, int, long，等任何Number或CharSequence（存储的是数字）子类型 | 验证注解的元素值大于等于@Min指定的value值                    |
| @Max（value=值）                             | 和@Min要求一样                                               | 验证注解的元素值小于等于@Max指定的value值                    |
| @DecimalMin(value=值)                        | 和@Min要求一样                                               | 验证注解的元素值大于等于@ DecimalMin指定的value值            |
| @DecimalMax(value=值)                        | 和@Min要求一样                                               | 验证注解的元素值小于等于@ DecimalMax指定的value值            |
| @Digits(integer=整数位数, fraction=小数位数) | 和@Min要求一样                                               | 验证注解的元素值的整数位数和小数位数上限                     |
| @Size(min=下限, max=上限)                    | 字符串、Collection、Map、数组等                              | 验证注解的元素值的在min和max（包含）指定区间之内，如字符长度、集合大小 |
| @Past                                        | java.util.Date,java.util.Calendar;Joda Time类库的日期类型    | 验证注解的元素值（日期类型）比当前时间早                     |
| @Future                                      | 与@Past要求一样                                              | 验证注解的元素值（日期类型）比当前时间晚                     |
| @NotBlank                                    | CharSequence子类型                                           | 验证注解的元素值不为空（不为null、去除首位空格后长度为0），不同于@NotEmpty，@NotBlank只应用于字符串且在比较时会去除字符串的首位空格 |
| @Length(min=下限, max=上限)                  | CharSequence子类型                                           | 验证注解的元素值长度在min和max区间内                         |
| @NotEmpty                                    | CharSequence子类型、Collection、Map、数组                    | 验证注解的元素值不为null且不为空（字符串长度不为0、集合大小不为0） |
| @Range(min=最小值, max=最大值)               | BigDecimal,BigInteger,CharSequence, byte, short, int, long等原子类型和包装类型 | 验证注解的元素值在最小值和最大值之间                         |
| @Email(regexp=正则表达式,flag=标志的模式)    | CharSequence子类型（如String）                               | 验证注解的元素值是Email，也可以通过regexp和flag指定自定义的email格式 |
| @Pattern(regexp=正则表达式,flag=标志的模式)  | String，任何CharSequence的子类型                             | 验证注解的元素值与指定的正则表达式匹配                       |
| @Valid                                       | 任何非原子类型                                               | 指定递归验证关联的对象如用户对象中有个地址对象属性，如果想在验证用户对象时一起验证地址对象的话，在地址对象上加@Valid注解即可级联验证 |





