
from django.contrib import admin
from django.urls import path
from face_detect.views import home, camera,gen
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", home),
    path("camera", camera, name="camera"),
    path("face", gen, name="face"),
]

urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
