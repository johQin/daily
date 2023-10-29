# django

[在pycharm专业版中打开django项目](https://blog.csdn.net/weixin_41924879/article/details/109602645)

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

# [models](https://blog.csdn.net/happygjcd/article/details/102649947)

一个Models对应数据库的一张表，Django中Models以类的形式表现，它包含了一些基本字段以及数据的一些行为

我们只需要在类里面进行操作，就可以操作数据库，表，不需要直接使用SQL语句

我们通过创建类来创建数据表，所以对数据库的操作，都是对类与对类的对象进行操作，而不使用sql语句

ORM对象关系映射，实现了对象和数据库的映射，隐藏了数据访问的细节，不需要编写SQL语句

```bash
# 生成数据表
python manage.py makemigrations
```



# views



## 五大视图

| 动作：视图类             | 是否需要model                             | 是否需要form                                |
| ------------------------ | ----------------------------------------- | ------------------------------------------- |
| 增：CreateView           | 是<br />由于需要操作数据库所以都需要model | 是<br />需要对web前端提交过来的数据进行校验 |
| 删：DeleteView           | 是                                        | 否                                          |
| 改：UpdateView           | 是                                        | 是                                          |
| 查：ListView，DetailView | 是                                        | 否                                          |
| 模板视图：TemplateView   |                                           |                                             |

## Form

基本作用：对前端传来的数据进行校验。

```python
# 新建一个form.py文件

from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import EmailValidator, RegexValidator

def check_nick_name(value):
    name_list = ['sb', 'cnm', 'mmp']
    for item in name_list:
        if item in value:
            raise ValidationError('昵称中不能包含敏感词汇', 'sensitive_words')

# 创建Form表单类
class UserForm(forms.Form):
    nick_name = forms.CharField(
        required=True,
        max_length=16,
        min_length=6,
        validators=[
            # RegexValidator(r'[\w]{6-16}', '昵称格式不正确')
            check_nick_name
        ],
        error_messages={
            'required': '不能为空',
            'max_length': '最大长度为16字符',
            'min_length': '最小长度为6字符',
        }
    )
    phone = forms.CharField(
        required=True,
        validators=[
            RegexValidator(r'^1[35678][0-9]{9}$', '手机号格式不正确'),
            RegexValidator(r'^1[35678]\d{9}$', '手机号格式不正确')
        ],
        error_messages={
            'required': '不能为空'
        }
    )
    # email = forms.EmailField(
    #     required=True,
    #     validators=[
    #         EmailValidator('邮箱格式不正确')
    #     ],
    #     error_messages={
    #         'required': '不能为空',
    #     }
    # )
```

```python
# 在views里

from django.http import HttpResponse,JsonResponse
from http import HTTPStatus

from .form import UserForm

def downloadModel(request):
    if request.method == "GET":
        dataInfo = UserForm(request.GET)   # 实例一个 自定义表单类的对象
        if dataInfo.is_valid():             # 进行数据校验
            data = dataInfo.cleaned_data  # 获取校验后的数据
            print(data.get('nick_name'))
            return JsonResponse({'code': 200, 'success': True, 'message': '下载成功'})
        else:
            error = dataInfo.errors.get_json_data()
            print('error:{}'.format(error))
            return JsonResponse({'code':-1, 'success': False, 'message': error})
            # HttpResponse(json.dumps(data), content_type='application/json')
    else:
        return HttpResponse(status=HTTPStatus.METHOD_NOT_ALLOWED)
```



# log

1. [关于settings.py被执行了两次](https://blog.csdn.net/weixin_42539198/article/details/88841550)

2. [You have 18 unapplied migration(s). Your project may not work properly until you apply](https://blog.csdn.net/yuan2019035055/article/details/126721102)：程序初始运行时，报这个错。

3. [django中读取settings中的相关参数](https://blog.csdn.net/liukai6/article/details/100113928)

4. [Django中返回json数据](https://www.fengnayun.com/news/content/125642.html)

5. [【Django】中间件详解](https://blog.csdn.net/al6nlee/article/details/129510694)

6. [如何在Django实现logging+Middleware记录服务端API日志](https://www.zhihu.com/question/572708729)

7. [Django怎么获取get请求里面的参数](https://blog.csdn.net/au55555/article/details/80024375)

8. [python 中使用socketio](https://www.cnblogs.com/focusTech/p/14595610.html)

9. [python manage.py subcommand [options]和 django-admin subcommand [options]](https://www.django.cn/article/show-26.html)

   - 在DJango里django-admin.py和manage.py都是Django的命令工具集，用于处理系统管理相关操作，而manage.py是在创建Django工程时自动生成的，manage.py是对django-admin.py的简单包装，二者的作用基本一致。

   - [参考](https://docs.djangoproject.com/zh-hans/4.2/ref/django-admin/)

   - ```bash
     # 常用子命令：
     startproject:创建一个项目（*）
     startapp:创建一个app（*）
     runserver：运行开发服务器（*）
     shell：进入django shell（*）
     dbshell：进入django dbshell
     check：检查django项目完整性
     flush：清空数据库
     compilemessages：编译语言文件
     makemessages：创建语言文件
     makemigrations：生成数据库同步脚本（*）
     migrate：同步数据库（*）
     showmigrations：查看生成的数据库同步脚本（*）
     sqlflush：查看生成清空数据库的脚本（*）
     sqlmigrate：查看数据库同步的sql语句（*）
     dumpdata:导出数据
     loaddata:导入数据
     diffsettings:查看你的配置和django默认配置的不同之处
     manage.py特有的一些子命令：
     createsuperuser:创建超级管理员（*）
     changepassword:修改密码（*）
     clearsessions：清除session
     ```

10. [Django的INSTALLED_APPS中应该写app名，还是AppConfig子类?](https://blog.csdn.net/bocai_xiaodaidai/article/details/113740272)

    - django3.2版本之后，INSTALLED_APPS只需要写app名称就可以了，django会自动寻找app.py中自定义的Config类加载，找不到则会使用Django默认的AppConfig。

    - ```python
      # settings.py 注册app
      # 方式1：直接加入app名，
      INSTALLED_APPS = [
          'django.contrib.admin',
          'app01',
      ]
       
      # django3.2 之前的版本 
      # 方式2：直接加入app对应的AppConfig子类
      INSTALLED_APPS = [
          'django.contrib.admin',
          'app01.apps.App01Config',
      ]
      ```

11. [Django 使用 gevent（或 eventlet）和 prefork worker 与 Celery](https://geek-docs.com/django/django-questions/176_django_using_both_gevent_or_eventlet_and_prefork_workers_with_celery.html)

12. [setting.py 中的WSGI_APPLICATION](https://geek-docs.com/django/django-questions/772_django_why_do_we_have_to_provide_wsgi_application_variable_in_django_settings.html)

13. [seting.py 中的INSTALLED_APPS](https://geek-docs.com/django/django-questions/457_django_what_does_installed_apps_setting_in_django_actually_do.html?action=all)

14. [断点下载](https://blog.csdn.net/wsfsp_4/article/details/127019804)

15. [Django+Celery+Flower实现异步和定时任务及其监控告警](https://xiejava.blog.csdn.net/article/details/128500555)

# Celery

用Django框架进行web开发非常的快捷方便，但Django框架请求/响应是同步的。但我们在实际项目中经常会碰到一些耗时的不能立即返回请求结果任务如：数据爬取、发邮件，下载大文件等，如果常时间等待对用户体验不是很好，在这种情况下就需要实现异步实现，马上返回响应请求，但真正的耗时任务在后台异步执行。Django框架本身无法实现异步响应但可以通过Celery很快的实现异步和定时任务。

常见的任务有两类：

- 异步任务，将耗时的操作任务提交给Celery去异步执行，比如发送短信/邮件、消息推送、音频处理等等
- 定时任务（定时执行或按一定周期执行）。





