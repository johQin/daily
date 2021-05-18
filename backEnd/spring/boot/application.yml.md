# application.yml

# 1 [dataSource](https://blog.csdn.net/qq_36803325/article/details/89574789)

| 属性                                    | 说明                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| spring.datasource.jdbc-url              | 用来创建连接的JDBC URL                                       |
| spring.datasource.username              | 数据库的登录用户名                                           |
| spring.datasource.password              | 数据库登录密码                                               |
| spring.datasource.pool-name             | 连接池名称                                                   |
| spring.datasource.driver-class-name     | JDBC驱动的全限定类名。默认根据URL自动检测                    |
| spring.datasource.maximum-pool-size     | 连接池能达到的最大规模，包含空闲的连接数量和使用中的连接数量 |
| spring.datasource.login-timeout         | 连接数据库的超时时间（单位为秒）                             |
| spring.datasource.max-lifetime          | 连接池中连接的最长寿命（单位为毫秒）                         |
| spring.datasource.max-wait              | 连接池在等待返回连接时，最长等待多少毫秒再抛出异常           |
| spring.datasource.max-active            | 连接池中最大活跃连接数                                       |
| spring.datasource.max-idle              | 连接池中最大空闲连接数                                       |
| spring.datasource.connection-timeout    | 连接超时时间（单位为毫秒）                                   |
| spring.datasource.connection-test-query | 用于测试连接有效的SQL查询                                    |
| spring.datasource.data-source-jndi      | 用户获取连接的数据源JNDI的位置                               |
| spring.datasource.min-idle              | 连接池里始终应该保持的最小连接数（用于DBCP和Tomcat连接池）   |
| spring.datasource.idle-timeout          | 连接池中连接能保持闲置状态的最长时间单位为毫秒（默认值为10） |
| spring.datasource.initial-size          | 在连接池启动的时候要建立的连接数                             |
| spring.datasource.test-on-connetct      | 在建立连接时是否要进行测试                                   |
| spring.datasource.jndi-name             | 数据源JNDI的位置，设置了该属性则忽略类、URL、用户名和密码    |
| spring.datasource.test-while-idle       | 在连接空闲时是否要进行测试                                   |

# 3 [server](<https://www.cnblogs.com/austinspark-jessylu/p/8065215.html>)

servlet，session，cookie，ssl，tomcat，undertow

# 2 JPA

| 属性                                 | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| spring.jpa.database                  | 指定目标数据库.                                              |
| spring.jpa.database-platform         | 指定目标数据库的类型                                         |
| spring.jpa.hibernate.ddl-auto        | 当使用内嵌数据库时，默认是create-drop，否则为none.           |
| spring.jpa.generate-ddl              | 是否在启动时初始化schema，默认为false                        |
| spring.jpa.hibernate.naming-strategy | 指定命名策略.                                                |
| spring.jpa.open-in-view              | 是否注册OpenEntityManagerInViewInterceptor，绑定JPA EntityManager到请求线程中，默认为: true |
| spring.jpa.properties                | 添加额外的属性到JPA provider.                                |
| spring.jpa.show-sql                  | 是否开启sql的log，默认为: false                              |

1. spring.jpa.hibernate.ddl-auto
   - ddl-auto:create----每次运行该程序，没有表格会新建表格，表内有数据会清空
   - ddl-auto:create-drop----每次程序结束的时候会清空表
   - ddl-auto:update----每次运行程序，没有表格会新建表格，表内有数据不会清空，只会更新
   - ddl-auto:validate----运行程序会校验数据与数据库的字段类型是否相同，不同会报错
2. show-sql
   - 控制台是否显示sql