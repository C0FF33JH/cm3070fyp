# Create your views here.
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from dashboard.models import Transaction, Account
from dashboard.ml_services import get_categorization_service
import json

def dashboard_view(request):
    """Main dashboard view"""
    # Default to demo_user for now (no login required)
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    # Get recent transactions
    transactions = Transaction.objects.filter(
        account=account,
        data_source='plaid'
    ).order_by('-date')[:50]
    
    # Get category statistics
    categorized = transactions.filter(ai_category__isnull=False).count()
    uncategorized = transactions.filter(ai_category__isnull=True).count()
    
    context = {
        'account': account,
        'transactions': transactions,
        'categorized': categorized,
        'uncategorized': uncategorized,
    }
    
    return render(request, 'dashboard/dashboard.html', context)

@require_http_methods(["POST"])
def categorize_transactions(request):
    """API endpoint to trigger categorization"""
    # Default to demo_user for now
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    service = get_categorization_service()
    
    # Categorize uncategorized transactions
    count = service.categorize_uncategorized_transactions(account.id)
    
    # Get updated statistics
    stats = service.get_category_statistics(account.id)
    
    return JsonResponse({
        'success': True,
        'categorized': count,
        'statistics': stats
    })

def category_stats_api(request):
    """API endpoint for category statistics"""
    # Default to demo_user for now
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    service = get_categorization_service()
    stats = service.get_category_statistics(account.id)
    
    return JsonResponse(stats)