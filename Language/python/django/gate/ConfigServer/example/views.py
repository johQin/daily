from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.views.generic import TemplateView

def startModel(request,*arg,**kwargs):
    # kwargs可以获取路由上匹配的动态路由参数{'num':1}
    # 而路由上的查询参数可以从request.GET中获取
    # request里面有很多信息，可以查看
    return HttpResponse('startModel')

def backTemplate(request,*arg,**kwargs):
    return TemplateResponse(request, template='index.html')
    # 要做template页面返回，先要在全局管理app——ConfigServer的setting里注册当前的appConfig（eg:此的ModelConfig）
    # 其次需要在该app建一个Templates包，template参数是相对于这个文件夹的路径来的。

class TemplateClass(TemplateView):
    template_name = 'classTemplate.html'
    def get_context_data(self, **kwargs):
        return {'world': '模板语法'}