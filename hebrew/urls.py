from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('letter_recognize/', include('letter_recognize.urls')),
    path('drawletters/', include('drawletters.urls')),
    path('admin/', admin.site.urls),
]