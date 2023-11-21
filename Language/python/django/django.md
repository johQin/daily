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

   ```python
   # settings.py
   IP_LOCAL = ["192.168.66.55", "169.254.111.198"]
   # views.py
   from django.conf import settings
   print(settings.IP_LOCAL)
   ```

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

16. [django中出现Forbidden (CSRF cookie not set.)](https://blog.csdn.net/g1655/article/details/117772715)

    ```python
    # 法一：在setting中注释下面这一行
    'django.middleware.csrf.CsrfViewMiddleware'
    # 法二：在view的方法上使用@csrf_exempt
    from django.views.decorators.csrf import csrf_exempt
    @csrf_exempt
    def demo():
        pass
    ```

17. 如果报这个警告：[UnorderedObjectListWarning: Pagination may yield inconsistent results with an unordered object_list](https://blog.csdn.net/time_money/article/details/122197913)

    ```python
    Entity.objects.filter(**filterParams).order_by("id")
    ```

18. 查询列表返回的部分字段并且列表里的对象要是字典

    - [values()返回对象为字典](https://www.cnblogs.com/regit/p/16825159.html)
    - [querySet转list](https://blog.51cto.com/u_16175486/6880227)

    ```python
    MyModel.objects.values()
    
    查询部分列的数据并返回，等同于select 列1，列2 from table
    返回值：QuerySet容器对象，内部存放字典， 每个字典代表一条数据；格式为{‘列1’:值1,‘列2’:值2}
    
    MyModel.objects.values_list()
    
    作用：返回元组形式的查询结果，等同于select 列1，列2 from xxx
    返回值：QuerySet容器对象，内部存放元组， 会将查询出来的数据封装到元组中，再封装到查询集合QuerySet中， 如果需要将查询结果取出，需要使用索引的方式取值
    ```

    ```python
    querySet = Entity.objects.filter(**filterParams).values('modelVersion','modelName').order_by("id")
    # values如果不填字段名，就查询表中所有字段，并且它将列表中的所有对象，转化字典的形式存储。
    # 返回的是QuerySet容器对象。
    
    # 转换QuerySet容器对象为list对象
    # 步骤1: 获取查询集
    queryset = Model.objects.all()
    
    # 步骤2: 将查询集转换为字典列表
    list_data = list(queryset.values())
    
    # 步骤3: 将查询集转换为对象列表
    # list_data = list(queryset)  # 如果你希望得到一个对象列表
    ```

19. django在使用mysql的时候，原来的字段为"textField"，并且已经有记录里放的是字符串，后面又修改为"jsonField", 这时候在使用migrate就会报错

    - `django.db.utils.OperationalError: (3140, 'Invalid JSON text: "Invalid value." at position 0 in value for column \'#sql-7f5a7_79.entityJson\'.')`
    - 这是因为原有记录里的字符串不是json格式的，所以导致数据库历史记录，无法转换为json而报错

20. [django中使用mysql，在model中指定表名](https://deepinout.com/mysql/mysql-questions/427_mysql_database_table_names_with_django.html)

    ```python
    # 在Django中，我们可以通过设置模型的db_table属性来控制表名
    # models.py
    class Article(models.Model):
        title = models.CharField(max_length=200)
        content = models.TextField()
    
        class Meta:
            db_table = 'article_table'
    ```

21. [在mysql中使用uuid做主键](https://deepinout.com/django/django-questions/418_django_implementing_uuid_as_primary_key.html)

    - [参考2](https://www.jianshu.com/p/62336d9a39ed)

    ```python
    from django.db import models
    import uuid
    
    class Author(models.Model):
        id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
        name = models.CharField(max_length=100)
        # 其他字段...
    
    class BlogPost(models.Model):
        id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
        title = models.CharField(max_length=200)
        content = models.TextField()
        created_at = models.DateTimeField(auto_now_add=True)
        author = models.ForeignKey(Author, on_delete=models.CASCADE)
    
        def __str__(self):
            return self.title
    # 在关联查询的时候
    在关联模型中使用 UUID 进行查询时，我们需要使用双下划线 __。
    author = Author.objects.get(blogpost__id='UUID值')
    ```

22. [多表连接查询](https://blog.51cto.com/u_16213452/7783350)

    ```python
    # models.py
    from django.db import models
    
    class Customer(models.Model):
        customer_name = models.CharField(max_length=100)
    
    class Order(models.Model):
        order_date = models.DateField()
        customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
       
    # views.py
    orders = Order.objects.select_related('customer')
    for order in orders:
        print(f"Order ID: {order.id}, Order Date: {order.order_date}, Customer Name: {order.customer.customer_name}")
    ```

23. [在view中使用事务](https://blog.csdn.net/CSDN1csdn1/article/details/133840978)

    - [参考2](https://blog.csdn.net/momoda118/article/details/128177420)

    ```python
    # 如果为了图省事而场景也较为通用，我们可以设置全局事务配置来让每个请求view都使用事务：
    # 在settings.py配置文件中增加下面配置：
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'testdb', 
            'USER': 'root',  
            'PASSWORD': '123', 
            'HOST': '127.0.0.1',     
            'PORT': 3306,  
            'ATOMIC_REQUESTS': True  # 全局开启事务，和http请求的整个过程绑定在一起
        }
    }
    
    ```

    

24. [django 批量新增和修改](https://www.cnblogs.com/luck-pig/p/17407847.html)

    ```python
    # 批量新增
    # 1. 定义一个 model对象的列表，
    tmpList=[]
    # 2. 用循环将需要创建的model对象，添加到此列表，
    tmpList.append(obj)
    # 3. 使用django提供的bulk_create方法将此对象列表填充，即可实现sql中的批量添加
    Obj.objects.bulk_create(tmpList)
    
    # 批量修改
    # 1. 定义一个 model对象的列表，
    tmpList=[]
    # 2. 用循环将需要创建的model对象，添加到此列表，
    tmpList.append(obj)
    # 3. 使用django提供的bulk_update方法将此对象列表填充，并且需要显示定义fields字段，更新的字段都要显示体现，即可实现sql中的批量添加
    Obj.objects.bulk_update(tmpList，fields=["xx"])
    ```

    

25. [外键](https://blog.51cto.com/u_16213452/7783350)

    ```python
    # 使用ForeignKey的字段所在的表定义为从表，把ForeignKey中to参数连接的表称为主表。
    
    class TargetModel(models.Model):
        tid = models.AutoField(primary_key=True)
        entity = models.ForeignKey(EntityModel, on_delete=models.CASCADE)		# EntityModel是主表
        gid = models.CharField(max_length=10)
        targetJson = models.JSONField("模型分析目标的参数配置")
    @require_POST
    def queryTargetListView(request):
        requestData = TargetQueryForm(json.loads(request.body))
        if requestData.is_valid():
            queryData = {k: v for k, v in requestData.cleaned_data.items() if requestData.cleaned_data[k]}
            filterRes = TargetModel.objects.filter(entity_id=queryData['modelId']).select_related('entity').order_by('tid')
            res = []
            if not filterRes.exists():
                return JsonResponse({'code': 200, 'success': True, 'message': '查询成功', 'data':res})
            entityJson = filterRes[0].entity.entityJson			# 依据外键获取主表对象的数据
            res['commonParams'] = entityJson['commonParams']
            modelType = filterRes[0].entity.modelType
            modelTypeParams = f'{modelType}Params'
            if modelTypeParams in entityJson:
                res[modelTypeParams] = entityJson[modelTypeParams]
            targets = []
            for t in filterRes.values():					# 将QuerySet转为dict 数组
                targets.append(t['targetJson'])
            res['list'] = targets
            return JsonResponse({'code': 200, 'success': True, 'message': '查询成功', 'data': res})
    
        else:
            error = requestData.errors.get_json_data()
            return JsonResponse({'code': -1, 'success': False, 'message': error})
       
    ```

26. [django.core.exceptions.AppRegistryNotReady: Apps aren‘t loaded yet.](https://blog.csdn.net/seanyang_/article/details/132632165)

    ```python
    # django.core.exceptions.AppRegistryNotReady: Apps aren‘t loaded yet.
    import django
    django.setup()
    # django.setup()`是Django框架中的一个函数。它用于在非Django环境下使用Django的各种功能、模型和设置。
    # 在常规的Django应用程序中，不需要手动调用`django.setup()`。Django在启动应用程序时会自动调用它来设置所需的环境和配置。
    ```

27. django.db.utils.OperationalError: (1054, "Unknown column 'model_entity.modelChineseName' in 'field list'")

    - 在新增字段的时候出现这样的问题，即使删除migrations，问题同样存在，这时，你就在mysql数据库中手动添加，然后再通过命令行去makemigrations和migrate

28. [django环境移植，安装mysqlclient包](https://blog.csdn.net/qwe1314225/article/details/132150159)

    ```bash
    # mysqlclient，主机缺少mysqlclient，
    # 在安装mysqlclient pip包，它依赖主机上的mysqlclient，所以如果如果主机上没有mysqlclient，那么就会报错，报错如下
    Pip subprocess error:
      error: subprocess-exited-with-error
      
      × Getting requirements to build wheel did not run successfully.
      │ exit code: 1
      ╰─> [22 lines of output]
          Trying pkg-config --exists mysqlclient
          Command 'pkg-config --exists mysqlclient' returned non-zero exit status 1.
          Trying pkg-config --exists mariadb
          Command 'pkg-config --exists mariadb' returned non-zero exit status 1.
          Traceback (most recent call last):
            File "/root/anaconda3/envs/djg_conf_server/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 353, in <module>
              main()
            File "/root/anaconda3/envs/djg_conf_server/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 335, in main
              json_out['return_val'] = hook(**hook_input['kwargs'])
            File "/root/anaconda3/envs/djg_conf_server/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 118, in get_requires_for_build_wheel
              return hook(config_settings)
            File "/tmp/pip-build-env-ttcgnh3w/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 325, in get_requires_for_build_wheel
              return self._get_build_requires(config_settings, requirements=['wheel'])
            File "/tmp/pip-build-env-ttcgnh3w/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 295, in _get_build_requires
              self.run_setup()
            File "/tmp/pip-build-env-ttcgnh3w/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 311, in run_setup
              exec(code, locals())
            File "<string>", line 154, in <module>
            File "<string>", line 48, in get_config_posix
            File "<string>", line 27, in find_package_name
          Exception: Can not find valid pkg-config name.
          Specify MYSQLCLIENT_CFLAGS and MYSQLCLIENT_LDFLAGS env vars manually
          [end of output]
      
      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: subprocess-exited-with-error
    
    × Getting requirements to build wheel did not run successfully.
    │ exit code: 1
    ╰─> See above for output.
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    
    failed
    
    CondaEnvException: Pip failed
    
    
    # 解决
    apt-get install libmysqlclient-dev
    apt-get install python3-dev
    ```

    

29. [通过中间件记录view的日志信息](https://blog.csdn.net/qq_42717671/article/details/132082232)

30. 

# 部署

```bash
/home/buntu/.conda/envs/djg_conf_server/bin/python /home/buntu/gitRepository/axxt/ModelDeployment/manage.py runserver 192.168.101.163:8000 --noreload
```



# Celery

用Django框架进行web开发非常的快捷方便，但Django框架请求/响应是同步的。但我们在实际项目中经常会碰到一些耗时的不能立即返回请求结果任务如：数据爬取、发邮件，下载大文件等，如果常时间等待对用户体验不是很好，在这种情况下就需要实现异步实现，马上返回响应请求，但真正的耗时任务在后台异步执行。Django框架本身无法实现异步响应但可以通过Celery很快的实现异步和定时任务。

常见的任务有两类：

- 异步任务，将耗时的操作任务提交给Celery去异步执行，比如发送短信/邮件、消息推送、音频处理等等
- 定时任务（定时执行或按一定周期执行）。

```bash
celery -A ModelDeployment worker -l debug -P eventlet
```



# Mysql

1. 安装依赖包，官方建议使用mysqlclient

   ```bash
   pip install mysqlclient
   ```

2. 配置文件

   ```python
   # settings.py
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.mysql',  # 数据库引擎
           'NAME': 'ModelDeployment',  # 数据库名称
           'HOST': '127.0.0.1',  # 数据库地址，本机 ip 地址 127.0.0.1
           'PORT': 3306,  # 端口
           'USER': 'root',  # 数据库用户名
           'PASSWORD': 'root',  # 数据库密码
       }
   }
   
   INSTALL_APPS = [
       ...
   ]
   ```

   

3. models orm的类

   ```python
   # app的models.py
   from django.db import models
   class Entity(models.Model):
       model_name= models.CharField(max_length=32,verbose_name='模型名称',help_text='帮助信息')
       model_version = models.CharField(max_length=10)
       start_cmd = models.CharField(max_length=512)
       common_json = models.TextField()
       entity_json = models.TextField()
   
       def __str__(self):
           return f"{self.id} {self.model_name} {self.model_version}"
   ```

4. 同步到数据库

   ```bash
   # 这里的python一定要写绝对地址，如果只写python或者使用django-admin那么会报，因为命令行中的命令有可能会连接到其它python解释器上去
   # raise ImproperlyConfigured(django.core.exceptions.ImproperlyConfigured: Requested setting LANGUAGES, but settings are not configured. You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() before accessing settings.
   
   /home/buntu/.conda/envs/djg_conf_server/bin/python manage.py makemigrations
   # 如果你的app下没有migrations包，你需要指定app的名字
   /home/buntu/.conda/envs/djg_conf_server/bin/python manage.py makemigrations ModelEntity
   
   # 在app下生成migrations包后
   # 再执行migrate，就会将类的变化同步到数据库中
   /home/buntu/.conda/envs/djg_conf_server/bin/python manage.py migrate
   ```

   - 注意

   - ```
     在进行迁移时，Django会遍历INSTALLED_APPS中列出的所有应用程序，查看现有迁移以构建迁移状态，然后将其与应用程序models模块中可用模型的当前状态进行比较。
     为了让它们被识别出来，你需要将它们都放在models.py中，或者，如果你使用一个包（models/目录中有多个文件），你需要将它们都导入到models/__init__.py文件中。
     当应用程序的目录中已经存在migrations/包时，makemigrations将只拾取模型并生成新的迁移。但是，如果你没有migrations包，Django不会自动拾取它，你必须显式指定应用程序的名称。
     例如，如果您安装了API.Peak应用程序，但没有API/Peak/migrations/文件夹，则必须明确提供该文件夹，即：python manage.py makemigrations Peak（注意，当应用程序中有点时，您只需指定最后一部分，因此API.Peak和API.Waterfall将仅指定为Peak或Waterfall）
     ```

5. [操作数据库](https://blog.csdn.net/m0_64599482/article/details/128100516)

# docker-py

```bash
pip install docker
```

[docker-py](https://blog.csdn.net/qq_42730750/article/details/128903132)

1. docker-py在使用脚本时

   - ```python
     client = docker.from_env()
     containers = client.containers.list(all=True, filters={'ancestor':'ubuntu:hello'})
     ```

   - 如果当前用户不是root，或者脚本不是通过sudo执行，它就会报`docker.errors.DockerException: Error while fetching server API version: (‘Connection aborted.‘, Permission...`

   - 这时需要将当前用户添加到docker用户组上去，并且添加完成之后，重启系统，既可生效

   - ```bash
     # docker 命令只能由 root 用户或 docker 组中的用户运行，该组是在 Docker 安装过程中自动创建的。如果您想避免在运行 docker 命令时输入 sudo，请将您的用户名添加到 docker 组
     sudo usermod -aG docker $USER
     # 添加之后，需要重启电脑然后才能生效
     ```

   - [参考1](https://blog.csdn.net/m0_57236802/article/details/131642832)

2. 在docker中使用container exec_run如果要后台运行，命令字符串里就不能使用nohup &，转而使用`detach=True`

   ```python
   container.exec_run("command", detach=True)
   ```

3. 

# Django-Ninja
