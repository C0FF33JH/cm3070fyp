from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.utils import timezone
from dashboard.models import Transaction, Account, ChatMessage, AnalysisResult
from dashboard.ml_services import get_categorization_service, get_forecasting_service, get_fraud_service
import json

@csrf_exempt
@require_http_methods(["POST"])
def parse_receipt(request):
    """API endpoint for receipt parsing"""
    from dashboard.ml_services.receipt_service import get_receipt_service
    
    # Check if image file was uploaded
    if 'receipt' not in request.FILES:
        return JsonResponse({'error': 'No receipt image provided'}, status=400)
    
    receipt_file = request.FILES['receipt']
    
    # Get user and account
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    # Parse receipt
    service = get_receipt_service()
    result = service.process_receipt(receipt_file)
    
    if result['success']:
        # Try to match to transaction
        receipt_data = result['data']
        if receipt_data.get('total'):
            match = service.match_receipt_to_transaction(receipt_data, account.id)
            result['matched_transaction'] = match
    
    return JsonResponse(result)

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

def forecast_cashflow(request):
    """API endpoint for cash flow forecasting"""
    # Default to demo_user for now
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    # Get parameters from request
    horizon = int(request.GET.get('days', 7))
    history = int(request.GET.get('history', 90))
    samples = int(request.GET.get('samples', 20))
    
    # Get forecasting service
    service = get_forecasting_service()
    
    # Generate forecast
    result = service.forecast_balance(
        account_id=account.id,
        horizon=horizon,
        history_days=history,
        num_samples=samples
    )
    
    # Add spending patterns if requested
    if request.GET.get('include_patterns') == 'true':
        result['spending_patterns'] = service.get_spending_patterns(account.id)
    
    return JsonResponse(result)

def detect_fraud(request):
    """API endpoint for fraud detection"""
    # Check which user type to analyze
    user_type = request.GET.get('user', 'fraud_analyst')
    
    if user_type == 'demo_user':
        # Statistical anomaly detection for Plaid data
        user = User.objects.get(username='demo_user')
        account = user.accounts.first()
        
        if not account:
            return JsonResponse({'error': 'No account found'}, status=404)
        
        sensitivity = float(request.GET.get('sensitivity', 2.0))
        service = get_fraud_service()
        result = service.detect_anomalies_plaid(account.id, sensitivity)
        
    else:
        # XGBoost fraud detection for Kaggle data
        user = User.objects.get(username='fraud_analyst')
        
        service = get_fraud_service()
        result = service.analyze_batch()
    
    return JsonResponse(result)

"""
Chat Interface Views
"""

def chat_interface(request):
    """Main chat interface view"""
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    # Get recent chat messages
    recent_messages = ChatMessage.objects.filter(
        user=user
    ).order_by('-created_at')[:10]
    
    context = {
        'account': account,
        'recent_messages': recent_messages,
    }
    
    return render(request, 'dashboard/chat.html', context)

@require_http_methods(["POST"])
def process_chat(request):
    """Process chat message through orchestration"""
    from dashboard.ml_services.orchestration_service import get_orchestration_service
    from dashboard.models import ChatMessage
    
    # Get user and account
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    # Get message from request
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
    except:
        user_message = request.POST.get('message', '').strip()
    
    if not user_message:
        return JsonResponse({'error': 'No message provided'}, status=400)
    
    # Get orchestration service
    service = get_orchestration_service()
    
    # Process the query
    response = service.process_query(
        user_query=user_message,
        account_id=account.id,
        context={
            'user_id': user.id,
            'timestamp': timezone.now().isoformat()
        }
    )
    
    # Save to chat history
    ChatMessage.objects.create(
        user=user,
        message=user_message,
        response=response
    )
    
    return JsonResponse(response)

def comprehensive_analysis(request):
    """Run all ML models for comprehensive analysis"""
    from dashboard.ml_services import (
        get_categorization_service,
        get_forecasting_service,
        get_fraud_service,
        get_orchestration_service
    )
    
    # Get user and account
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found'}, status=404)
    
    results = {
        'account_id': account.id,
        'account_name': account.account_name,
        'timestamp': timezone.now().isoformat(),
        'analyses': {}
    }
    
    # 1. Run categorization
    try:
        cat_service = get_categorization_service()
        cat_stats = cat_service.get_category_statistics(account.id)
        results['analyses']['categorization'] = {
            'success': True,
            'data': cat_stats
        }
    except Exception as e:
        results['analyses']['categorization'] = {
            'success': False,
            'error': str(e)
        }
    
    # 2. Run forecasting
    try:
        forecast_service = get_forecasting_service()
        forecast_result = forecast_service.forecast_balance(
            account.id,
            horizon=7,
            history_days=90
        )
        results['analyses']['forecast'] = {
            'success': forecast_result.get('success', False),
            'data': forecast_result
        }
    except Exception as e:
        results['analyses']['forecast'] = {
            'success': False,
            'error': str(e)
        }
    
    # 3. Run anomaly detection
    try:
        fraud_service = get_fraud_service()
        anomalies = fraud_service.detect_anomalies_plaid(
            account.id,
            sensitivity=2.0
        )
        results['analyses']['anomalies'] = {
            'success': True,
            'data': anomalies
        }
    except Exception as e:
        results['analyses']['anomalies'] = {
            'success': False,
            'error': str(e)
        }
    
    # 4. Generate insights summary
    orchestration = get_orchestration_service()
    summary_response = orchestration.process_query(
        "Give me a comprehensive analysis of my finances",
        account.id
    )
    results['summary'] = summary_response
    
    # Save as analysis result
    from dashboard.models import AnalysisResult
    AnalysisResult.objects.create(
        user=user,
        analysis_type='comprehensive',
        result_data=results
    )
    
    return JsonResponse(results)