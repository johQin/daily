from django.urls import path
from example.views import *
app_name = 'example'
urlpatterns = [
    path('start/<int:num>', startModel, name='index'),         #url: /model/start/1，num为int值映射的变量名，放在了函数的kwargs里
    path('backtemplate/',backTemplate,name='back'),
    path('classTemplate/', TemplateClass.as_view())

]