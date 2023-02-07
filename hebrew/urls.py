from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('drawletters/', include('drawletters.urls')),
    path('admin/', admin.site.urls),
]