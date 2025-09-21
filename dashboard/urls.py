"""
Dashboard URL Configuration
File: dashboard/urls.py
"""

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/categorize/', views.categorize_transactions, name='categorize'),
    path('api/category-stats/', views.category_stats_api, name='category_stats'),
    path('api/forecast/', views.forecast_cashflow, name='forecast'),
]