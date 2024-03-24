from django.urls import path
from classify_eye_disease import views

urlpatterns = [

    path('', views.index, name='index'),
]
