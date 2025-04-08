from django.urls import path
from plagiarism_checker import views

urlpatterns = [
    path('', views.home, name='home'),
    path('check/', views.check_plagiarism, name='check_plagiarism'),
]