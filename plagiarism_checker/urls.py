from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('check/', views.check_plagiarism, name='check_plagiarism'),
    path('results/', views.results_view, name='results'),
]