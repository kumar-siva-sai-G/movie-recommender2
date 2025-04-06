from django.urls import path
from . import views

urlpatterns = [
    path('templates/blogapp2/recommend/', views.recommend_view, name='recommend'),
]
