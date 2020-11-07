
from django.contrib import admin
from django.urls import path
from amine.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home2,name='home2'),
    path('<int:cluster_id>',home3, name='home3'),
]
