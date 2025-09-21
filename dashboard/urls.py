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
    path('api/fraud/', views.detect_fraud, name='fraud'),
    path('chat/', views.chat_interface, name='chat'),
    path('api/chat/', views.process_chat, name='process_chat'),
    path('api/comprehensive/', views.comprehensive_analysis, name='comprehensive'),
]