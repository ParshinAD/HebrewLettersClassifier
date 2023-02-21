from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('hook2', views.hook2, name='hook2'),
    path('hook3', views.hook3, name='hook3'),

]