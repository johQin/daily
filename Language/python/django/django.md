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
- template——模板引擎（对应于MVC的V），前后端分离的话，返回给前端json，让前端通过json数据，前端渲染模板。非必须
- views——视图（对应于MVC的C），必须

**浏览器输入地址——>urls.py——>view**——>model——>数据库——>**views**——>template——>response(字节码)——>**浏览器**

请求与响应

request——>urls（path）

response——>Views(httpResponse，TemplateResponse)