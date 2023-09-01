from django.urls import path
from person_info.views import *
app_name = 'person_info'
urlpatterns = [
    path('list/',PersonList.as_view(), name='person_list')
]