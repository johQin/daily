# django

```bash
# 全局管理app
django-admin startproject ConfigServer

├── ConfigServer
│   ├── asgi.py
│   ├── __init__.py
│   ├── settings.py		# 全局的配置
│   ├── urls.py			# 路由匹配
│   └── wsgi.py
└── manage.py

# 子app
django-admin startapp Model
```

MTV——model，template，view

- model——数据库管理员，也可以直接使用pymysql，非必须
- template——模板引擎（对应于MVC的V，view），前后端分离的话，返回给前端json，让前端通过json数据，前端渲染模板。非必须
- views——视图（对应于MVC的C，cotroller），必须

**浏览器输入地址——>urls.py——>view**——>model——>数据库——>**views**——>template——>response(字节码)——>**浏览器**

请求与响应

request——>urls（path）

response——>Views(httpResponse，TemplateResponse)

# app的结构

1. urls.py，路由转发
2. views.py，controller
3. models.py，orm

## [models](https://blog.csdn.net/happygjcd/article/details/102649947)

一个Models对应数据库的一张表，Django中Models以类的形式表现，它包含了一些基本字段以及数据的一些行为

我们只需要在类里面进行操作，就可以操作数据库，表，不需要直接使用SQL语句

我们通过创建类来创建数据表，所以对数据库的操作，都是对类与对类的对象进行操作，而不使用sql语句

ORM对象关系映射，实现了对象和数据库的映射，隐藏了数据访问的细节，不需要编写SQL语句

```bash
# 生成数据表

```



## views



### 五大视图

| 动作：视图类             | 是否需要model                             | 是否需要form                                |
| ------------------------ | ----------------------------------------- | ------------------------------------------- |
| 增：CreateView           | 是<br />由于需要操作数据库所以都需要model | 是<br />需要对web前端提交过来的数据进行校验 |
| 删：DeleteView           | 是                                        | 否                                          |
| 改：UpdateView           | 是                                        | 是                                          |
| 查：ListView，DetailView | 是                                        | 否                                          |
| 模板视图：TemplateView   |                                           |                                             |

