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
    # path('api/forecast/', views.forecast_cashflow, name='forecast'),
    path('api/forecast/', views.forecast_cashflow_api, name='forecast_api'), 
    path('api/fraud/', views.detect_fraud, name='fraud'),
    path('chat/', views.chat_interface, name='chat'),
    path('api/chat/', views.process_chat, name='process_chat'),
    path('api/comprehensive/', views.comprehensive_analysis, name='comprehensive'),
    path('api/parse-receipt/', views.parse_receipt, name='parse_receipt'),

    path('categorization/', views.categorization_view, name='categorization'),
    path('forecast/', views.forecast_view, name='forecast'),
    path('fraud/', views.fraud_view, name='fraud'),
]