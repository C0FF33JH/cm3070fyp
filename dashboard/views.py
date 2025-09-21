from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.utils import timezone
from dashboard.models import Transaction, Account, ChatMessage, AnalysisResult
from dashboard.ml_services import get_categorization_service, get_forecasting_service, get_fraud_service
import json
from django.db.models import Sum, Count, Avg
from decimal import Decimal
from datetime import datetime, date
import numpy as np

# Add this helper function at the top of your views.py (after imports)

def clean_for_json_field(data):
    """
    Recursively clean data for Django's JSONField storage
    Converts Decimal, datetime, and numpy types to JSON-serializable formats
    """
    import numpy as np
    from decimal import Decimal
    from datetime import datetime, date
    
    if isinstance(data, dict):
        return {k: clean_for_json_field(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json_field(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif hasattr(data, 'dtype'):  # numpy types
        if np.issubdtype(data.dtype, np.bool_):
            return bool(data)
        elif np.issubdtype(data.dtype, np.integer):
            return int(data)
        elif np.issubdtype(data.dtype, np.floating):
            return float(data)
    return data

# Custom JSON encoder to handle Decimal
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)
    

# The forecast page view
def forecast_view(request):
    """Cash flow forecast dashboard view"""
    # For demo, use demo_user
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    context = {
        'user': user,
        'account': account,
    }
    
    return render(request, 'dashboard/forecast.html', context)

# # The API endpoint - make sure this replaces the existing forecast_cashflow function
# def forecast_cashflow_api(request):
#     """API endpoint for cash flow forecasting"""
#     # Default to demo_user for now
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
#     # Get parameters from request
#     horizon = int(request.GET.get('days', 7))
#     history = int(request.GET.get('history', 90))
#     samples = int(request.GET.get('samples', 20))
    
#     try:
#         # Get forecasting service
#         from dashboard.ml_services import get_forecasting_service
#         service = get_forecasting_service()
        
#         # Generate forecast
#         result = service.forecast_balance(
#             account_id=account.id,
#             horizon=horizon,
#             history_days=history,
#             num_samples=samples
#         )
        
#         # Check if forecast was successful
#         if 'error' in result:
#             return JsonResponse({
#                 'success': False,
#                 'error': result['error']
#             })
        
#         # Add success flag
#         result['success'] = True
        
#         # Add spending patterns if requested
#         if request.GET.get('include_patterns') == 'true':
#             try:
#                 patterns = service.get_spending_patterns(account.id)
#                 # Convert any Decimal values in patterns
#                 if patterns:
#                     for key in patterns:
#                         if isinstance(patterns[key], Decimal):
#                             patterns[key] = float(patterns[key])
#                         elif isinstance(patterns[key], list):
#                             for item in patterns[key]:
#                                 if isinstance(item, dict):
#                                     for k, v in item.items():
#                                         if isinstance(v, Decimal):
#                                             item[k] = float(v)
#                 result['spending_patterns'] = patterns
#             except Exception as e:
#                 print(f"Error getting spending patterns: {e}")
#                 # Continue without patterns
        
#         # Ensure all numeric values are JSON serializable
#         return JsonResponse(result, encoder=DecimalEncoder)
        
#     except Exception as e:
#         print(f"Forecast error: {e}")
#         import traceback
#         traceback.print_exc()
#         return JsonResponse({
#             'success': False,
#             'error': str(e)
#         }, status=500)

# Also update the existing forecast_cashflow function to handle the DecimalEncoder:

# def forecast_cashflow(request):
#     """API endpoint for cash flow forecasting"""
#     # Default to demo_user for now
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return JsonResponse({'error': 'No account found'}, status=404)
    
#     # Get parameters from request
#     horizon = int(request.GET.get('days', 7))
#     history = int(request.GET.get('history', 90))
#     samples = int(request.GET.get('samples', 20))
    
#     # Get forecasting service
#     from dashboard.ml_services import get_forecasting_service
#     service = get_forecasting_service()
    
#     # Generate forecast
#     result = service.forecast_balance(
#         account_id=account.id,
#         horizon=horizon,
#         history_days=history,
#         num_samples=samples
#     )
    
#     # Add spending patterns if requested
#     if request.GET.get('include_patterns') == 'true':
#         patterns = service.get_spending_patterns(account.id)
#         # Convert any Decimal values in patterns
#         if patterns:
#             for key in patterns:
#                 if isinstance(patterns[key], Decimal):
#                     patterns[key] = float(patterns[key])
#                 elif isinstance(patterns[key], list):
#                     for item in patterns[key]:
#                         if isinstance(item, dict):
#                             for k, v in item.items():
#                                 if isinstance(v, Decimal):
#                                     item[k] = float(v)
#         result['spending_patterns'] = patterns
    
#     return JsonResponse(result, cls=DecimalEncoder)

# def categorization_view(request):
#     """Transaction categorization dashboard view"""
#     # For demo, use demo_user (remove this in production and use @login_required)
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return render(request, 'dashboard/no_account.html')
    
#     # Get category statistics
#     from dashboard.ml_services import get_categorization_service
#     service = get_categorization_service()
#     stats = service.get_category_statistics(account.id)
    
#     # Convert Decimal values in stats to float
#     if stats.get('categories'):
#         for cat in stats['categories']:
#             if 'total_amount' in cat and cat['total_amount']:
#                 cat['total_amount'] = float(cat['total_amount'])
#             if 'avg_amount' in cat and cat['avg_amount']:
#                 cat['avg_amount'] = float(cat['avg_amount'])
#             if 'avg_confidence' in cat and cat['avg_confidence']:
#                 cat['avg_confidence'] = float(cat['avg_confidence'])
    
#     if stats.get('average_confidence'):
#         stats['average_confidence'] = float(stats['average_confidence'])
    
#     # Get recent categorized transactions
#     recent_transactions = Transaction.objects.filter(
#         account=account,
#         ai_category__isnull=False,
#         data_source='plaid'
#     ).order_by('-date')[:20].values(
#         'transaction_id', 'date', 'description', 'merchant',
#         'amount', 'ai_category', 'confidence_score'
#     )
    
#     # Convert to JSON for JavaScript
#     transactions_list = []
#     for txn in recent_transactions:
#         transactions_list.append({
#             'id': txn['transaction_id'],
#             'date': txn['date'].strftime('%Y-%m-%d'),
#             'description': txn['description'],
#             'merchant': txn['merchant'] or '',
#             'amount': float(txn['amount']),  # Convert Decimal to float
#             'category': txn['ai_category'],
#             'confidence': float(txn['confidence_score'])
#         })
    
#     # Get counts
#     total_transactions = Transaction.objects.filter(
#         account=account,
#         data_source='plaid'
#     ).count()
    
#     categorized = Transaction.objects.filter(
#         account=account,
#         data_source='plaid',
#         ai_category__isnull=False
#     ).count()
    
#     uncategorized = total_transactions - categorized
    
#     context = {
#         'user': user,
#         'account': account,
#         'stats': stats,
#         'stats_json': json.dumps(stats, cls=DecimalEncoder),  # Use custom encoder
#         'transactions_json': json.dumps(transactions_list, cls=DecimalEncoder),
#         'total_transactions': total_transactions,
#         'categorized': categorized,
#         'uncategorized': uncategorized,
#         'avg_confidence': float(stats.get('average_confidence', 0)) if stats.get('average_confidence') else 0,
#     }
    
#     return render(request, 'dashboard/categorization.html', context)

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

# def forecast_cashflow(request):
#     """API endpoint for cash flow forecasting"""
#     # Default to demo_user for now
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return JsonResponse({'error': 'No account found'}, status=404)
    
#     # Get parameters from request
#     horizon = int(request.GET.get('days', 7))
#     history = int(request.GET.get('history', 90))
#     samples = int(request.GET.get('samples', 20))
    
#     # Get forecasting service
#     service = get_forecasting_service()
    
#     # Generate forecast
#     result = service.forecast_balance(
#         account_id=account.id,
#         horizon=horizon,
#         history_days=history,
#         num_samples=samples
#     )
    
#     # Add spending patterns if requested
#     if request.GET.get('include_patterns') == 'true':
#         result['spending_patterns'] = service.get_spending_patterns(account.id)
    
#     return JsonResponse(result)

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


# Update these in your dashboard/views.py

from decimal import Decimal
from datetime import datetime, date
import json

# Enhanced JSON encoder to handle both Decimal and datetime
# class CustomJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Decimal):
#             return float(obj)
#         elif isinstance(obj, (datetime, date)):
#             return obj.isoformat()
#         return super(CustomJSONEncoder, self).default(obj)
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Django and NumPy types"""
    def default(self, obj):
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle datetime/date
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle NumPy types
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle native Python bool (just in case)
        elif isinstance(obj, bool):
            return obj
        
        return super(CustomJSONEncoder, self).default(obj)

# Updated forecast API endpoint
def forecast_cashflow_api(request):
    """API endpoint for cash flow forecasting"""
    # Default to demo_user for now
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
    # Get parameters from request
    horizon = int(request.GET.get('days', 7))
    history = int(request.GET.get('history', 90))
    samples = int(request.GET.get('samples', 20))
    
    try:
        # Get forecasting service
        from dashboard.ml_services import get_forecasting_service
        service = get_forecasting_service()
        
        # Generate forecast
        result = service.forecast_balance(
            account_id=account.id,
            horizon=horizon,
            history_days=history,
            num_samples=samples
        )
        
        # Check if forecast was successful
        if 'error' in result:
            return JsonResponse({
                'success': False,
                'error': result['error']
            })
        
        # Add success flag
        result['success'] = True
        
        # Add spending patterns if requested
        if request.GET.get('include_patterns') == 'true':
            try:
                patterns = service.get_spending_patterns(account.id)
                result['spending_patterns'] = patterns
            except Exception as e:
                print(f"Error getting spending patterns: {e}")
                # Continue without patterns
        
        # Use the enhanced encoder
        return JsonResponse(result, encoder=CustomJSONEncoder, safe=False)
        
    except Exception as e:
        print(f"Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# Also update the categorization_view to use the new encoder
def categorization_view(request):
    """Transaction categorization dashboard view"""
    # For demo, use demo_user
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    # Get category statistics
    from dashboard.ml_services import get_categorization_service
    service = get_categorization_service()
    stats = service.get_category_statistics(account.id)
    
    # Convert Decimal values in stats to float
    if stats.get('categories'):
        for cat in stats['categories']:
            if 'total_amount' in cat and cat['total_amount']:
                cat['total_amount'] = float(cat['total_amount'])
            if 'avg_amount' in cat and cat['avg_amount']:
                cat['avg_amount'] = float(cat['avg_amount'])
            if 'avg_confidence' in cat and cat['avg_confidence']:
                cat['avg_confidence'] = float(cat['avg_confidence'])
    
    if stats.get('average_confidence'):
        stats['average_confidence'] = float(stats['average_confidence'])
    
    # Get recent categorized transactions
    recent_transactions = Transaction.objects.filter(
        account=account,
        ai_category__isnull=False,
        data_source='plaid'
    ).order_by('-date')[:20].values(
        'transaction_id', 'date', 'description', 'merchant',
        'amount', 'ai_category', 'confidence_score'
    )
    
    # Convert to JSON for JavaScript
    transactions_list = []
    for txn in recent_transactions:
        transactions_list.append({
            'id': txn['transaction_id'],
            'date': txn['date'].strftime('%Y-%m-%d'),
            'description': txn['description'],
            'merchant': txn['merchant'] or '',
            'amount': float(txn['amount']),
            'category': txn['ai_category'],
            'confidence': float(txn['confidence_score'])
        })
    
    # Get counts
    total_transactions = Transaction.objects.filter(
        account=account,
        data_source='plaid'
    ).count()
    
    categorized = Transaction.objects.filter(
        account=account,
        data_source='plaid',
        ai_category__isnull=False
    ).count()
    
    uncategorized = total_transactions - categorized
    
    context = {
        'user': user,
        'account': account,
        'stats': stats,
        'stats_json': json.dumps(stats, cls=CustomJSONEncoder),  # Use new encoder
        'transactions_json': json.dumps(transactions_list, cls=CustomJSONEncoder),
        'total_transactions': total_transactions,
        'categorized': categorized,
        'uncategorized': uncategorized,
        'avg_confidence': float(stats.get('average_confidence', 0)) if stats.get('average_confidence') else 0,
    }
    
    return render(request, 'dashboard/categorization.html', context)

# Add this view to dashboard/views.py

# def fraud_view(request):
#     """Fraud detection dashboard view"""
#     # Determine which mode to use based on available data
#     # Check if we have Kaggle data
#     from dashboard.models import Transaction
    
#     has_kaggle = Transaction.objects.filter(data_source='kaggle').exists()
    
#     if has_kaggle:
#         # Use fraud_analyst user for Kaggle data
#         user = User.objects.get(username='fraud_analyst')
#         data_mode = 'kaggle'
#     else:
#         # Use demo_user for statistical anomaly detection
#         user = User.objects.get(username='demo_user')
#         data_mode = 'demo'
    
#     account = user.accounts.first()
    
#     if not account:
#         # If no account, fallback to demo user
#         user = User.objects.get(username='demo_user')
#         account = user.accounts.first()
#         data_mode = 'demo'
    
#     context = {
#         'user': user,
#         'account': account,
#         'data_mode': data_mode,
#     }
    
#     return render(request, 'dashboard/fraud.html', context)
def fraud_view(request):
    """Fraud detection dashboard view"""
    # Always get demo_user for display
    demo_user = User.objects.get(username='demo_user')
    demo_account = demo_user.accounts.first()
    
    # Determine which mode to use based on available data
    from dashboard.models import Transaction
    
    has_kaggle = Transaction.objects.filter(data_source='kaggle').exists()
    
    if has_kaggle:
        data_mode = 'kaggle'
    else:
        data_mode = 'demo'
    
    context = {
        'user': demo_user,  # Keep for compatibility
        'demo_user': demo_user,  # Explicit demo user
        'demo_account': demo_account,  # Demo account for balance
        'account': demo_account,  # Keep for compatibility
        'data_mode': data_mode,
    }
    
    return render(request, 'dashboard/fraud.html', context)


# Also, let's ensure the detect_fraud function properly uses this encoder
def detect_fraud(request):
    """API endpoint for fraud detection"""
    # Check which user type to analyze
    user_type = request.GET.get('user', 'fraud_analyst')
    
    try:
        if user_type == 'demo_user':
            # Statistical anomaly detection for Plaid data
            user = User.objects.get(username='demo_user')
            account = user.accounts.first()
            
            if not account:
                return JsonResponse({'error': 'No account found'}, status=404)
            
            sensitivity = float(request.GET.get('sensitivity', 2.0))
            
            from dashboard.ml_services import get_fraud_service
            service = get_fraud_service()
            result = service.detect_anomalies_plaid(account.id, sensitivity)
            
        else:
            # XGBoost fraud detection for Kaggle data
            try:
                user = User.objects.get(username='fraud_analyst')
                # Check if user has accounts with Kaggle data
                from dashboard.models import Transaction
                has_kaggle = Transaction.objects.filter(data_source='kaggle').exists()
                
                if not has_kaggle:
                    # Fallback to demo mode
                    user = User.objects.get(username='demo_user')
                    account = user.accounts.first()
                    
                    if not account:
                        return JsonResponse({'error': 'No account found'}, status=404)
                    
                    sensitivity = float(request.GET.get('sensitivity', 2.0))
                    
                    from dashboard.ml_services import get_fraud_service
                    service = get_fraud_service()
                    result = service.detect_anomalies_plaid(account.id, sensitivity)
                else:
                    from dashboard.ml_services import get_fraud_service
                    service = get_fraud_service()
                    result = service.analyze_batch()
                    
            except User.DoesNotExist:
                # Fallback to demo user
                user = User.objects.get(username='demo_user')
                account = user.accounts.first()
                
                if not account:
                    return JsonResponse({'error': 'No account found'}, status=404)
                
                sensitivity = float(request.GET.get('sensitivity', 2.0))
                
                from dashboard.ml_services import get_fraud_service
                service = get_fraud_service()
                result = service.detect_anomalies_plaid(account.id, sensitivity)
        
        # Clean up the result to ensure all values are JSON serializable
        # This is a safety net in case the ML services return numpy types
        def clean_for_json(data):
            """Recursively clean data for JSON serialization"""
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            elif isinstance(data, (np.bool_, bool)):
                return bool(data)
            elif isinstance(data, (np.integer, int)):
                return int(data)
            elif isinstance(data, (np.floating, float)):
                return float(data)
            elif isinstance(data, (np.ndarray,)):
                return data.tolist()
            elif isinstance(data, Decimal):
                return float(data)
            elif isinstance(data, (datetime, date)):
                return data.isoformat()
            return data
        
        # Clean the result
        result = clean_for_json(result)
        
        return JsonResponse(result, encoder=CustomJSONEncoder, safe=False)
        
    except Exception as e:
        print(f"Fraud detection error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'error': str(e),
            'total_analyzed': 0,
            'anomalies_found': 0
        }, status=500)
    

# Update the chat_interface view in dashboard/views.py

def chat_interface(request):
    """Main chat interface view"""
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    # Get recent chat messages (optional)
    recent_messages = ChatMessage.objects.filter(
        user=user
    ).order_by('-created_at')[:10]
    
    context = {
        'user': user,
        'account': account,
        'recent_messages': recent_messages,
    }
    
    return render(request, 'dashboard/chat.html', context)

# Also ensure process_chat is using CustomJSONEncoder
# Then update your process_chat function - just replace the ChatMessage.create part:

# @require_http_methods(["POST"])
# def process_chat(request):
#     """Process chat message through orchestration"""
#     from dashboard.ml_services.orchestration_service import get_orchestration_service
#     from dashboard.models import ChatMessage
    
#     # Get user and account
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
#     # Get message from request
#     try:
#         data = json.loads(request.body)
#         user_message = data.get('message', '').strip()
#     except:
#         user_message = request.POST.get('message', '').strip()
    
#     if not user_message:
#         return JsonResponse({'error': 'No message provided', 'success': False}, status=400)
    
#     # Get orchestration service
#     service = get_orchestration_service()
    
#     try:
#         # Process the query
#         response = service.process_query(
#             user_query=user_message,
#             account_id=account.id,
#             context={
#                 'user_id': user.id,
#                 'timestamp': timezone.now().isoformat()
#             }
#         )
        
#         # Ensure success flag is set
#         if 'success' not in response:
#             response['success'] = True
        
#         # Clean the response for JSONField storage
#         cleaned_response = clean_for_json_field(response)
        
#         # Save to chat history with cleaned response
#         ChatMessage.objects.create(
#             user=user,
#             message=user_message,
#             response=cleaned_response  # Use cleaned version for storage
#         )
        
#         # Return the response (it will be properly encoded by CustomJSONEncoder)
#         return JsonResponse(response, encoder=CustomJSONEncoder, safe=False)
        
#     except Exception as e:
#         print(f"Chat processing error: {e}")
#         import traceback
#         traceback.print_exc()
        
#         return JsonResponse({
#             'success': False,
#             'error': str(e),
#             'natural_language_response': "I'm sorry, I encountered an error processing your request. Please try again."
#         }, status=500)

# Update the process_chat function in views.py with better categorization handling

@require_http_methods(["POST"])
def process_chat(request):
    """Process chat message through orchestration"""
    from dashboard.ml_services.orchestration_service import get_orchestration_service
    from dashboard.models import ChatMessage
    
    # Get user and account
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
    # Get message from request
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
    except:
        user_message = request.POST.get('message', '').strip()
    
    if not user_message:
        return JsonResponse({'error': 'No message provided', 'success': False}, status=400)
    
    # Get orchestration service
    service = get_orchestration_service()
    
    try:
        # Process the query
        response = service.process_query(
            user_query=user_message,
            account_id=account.id,
            context={
                'user_id': user.id,
                'timestamp': timezone.now().isoformat()
            }
        )
        
        # Check if we got a proper categorization response
        if 'categoriz' in user_message.lower():
            # If no categorization data in response, try direct approach
            if not response.get('data', {}).get('categorization'):
                print("No categorization in response, trying direct approach")
                from dashboard.ml_services import get_categorization_service
                cat_service = get_categorization_service()
                
                try:
                    # Run categorization
                    count = cat_service.categorize_uncategorized_transactions(account.id)
                    stats = cat_service.get_category_statistics(account.id)
                    
                    # Build proper response
                    categories = stats.get('categories', [])
                    spending_cats = [c for c in categories if c.get('total_amount', 0) < 0]
                    spending_cats.sort(key=lambda x: x.get('total_amount', 0))
                    
                    # Create response text
                    response_text = f"All {stats.get('total_transactions', 0)} transactions are already categorized. "
                    if spending_cats:
                        response_text += "Here's your spending breakdown:\n"
                        for i, cat in enumerate(spending_cats[:5], 1):
                            cat_name = cat['ai_category'].replace('_', ' ').title()
                            amount = abs(float(cat.get('total_amount', 0)))
                            response_text += f"\n{i}. {cat_name}: ${amount:,.2f}"
                    
                    # Create chart data
                    chart_data = {
                        'type': 'pie',
                        'title': 'Spending by Category',
                        'data': [
                            {
                                'category': cat['ai_category'].replace('_', ' ').title(),
                                'amount': abs(float(cat.get('total_amount', 0)))
                            }
                            for cat in spending_cats[:8]
                        ]
                    }
                    
                    # Build complete response
                    response = {
                        'success': True,
                        'natural_language_response': response_text,
                        'data': {
                            'categorization': {
                                'statistics': stats,
                                'newly_categorized': count,
                                'success': True
                            }
                        },
                        'charts': [chart_data],
                        'insights': [
                            f"📊 Total transactions: {stats.get('total_transactions', 0)}",
                            f"🎯 Average confidence: {(stats.get('average_confidence', 0) * 100):.1f}%"
                        ]
                    }
                    
                except Exception as e:
                    print(f"Direct categorization failed: {e}")
        
        # Ensure success flag is set
        if 'success' not in response:
            response['success'] = True
        
        # Clean the response for JSONField storage
        cleaned_response = clean_for_json_field(response)
        
        # Save to chat history with cleaned response
        ChatMessage.objects.create(
            user=user,
            message=user_message,
            response=cleaned_response
        )
        
        return JsonResponse(response, encoder=CustomJSONEncoder, safe=False)
        
    except Exception as e:
        print(f"Chat processing error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try direct categorization as last resort
        if 'categoriz' in user_message.lower():
            try:
                from dashboard.ml_services import get_categorization_service
                cat_service = get_categorization_service()
                stats = cat_service.get_category_statistics(account.id)
                
                categories = stats.get('categories', [])
                spending_cats = [c for c in categories if c.get('total_amount', 0) < 0]
                
                return JsonResponse({
                    'success': True,
                    'natural_language_response': f"You have {stats.get('total_transactions', 0)} categorized transactions.",
                    'data': {'categorization': {'statistics': stats}},
                }, encoder=CustomJSONEncoder, safe=False)
            except:
                pass
        
        return JsonResponse({
            'success': False,
            'error': str(e),
            'natural_language_response': "I'm sorry, I encountered an error processing your request. Please try again."
        }, status=500)
    
# Add these views to dashboard/views.py

def receipts_view(request):
    """Receipt parser dashboard view"""
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return render(request, 'dashboard/no_account.html')
    
    context = {
        'user': user,
        'account': account,
    }
    
    return render(request, 'dashboard/receipts.html', context)

# @csrf_exempt
# @require_http_methods(["POST"])
# def parse_receipt_api(request):
#     """API endpoint for receipt parsing"""
#     from dashboard.ml_services.receipt_service import get_receipt_service
    
#     # Check if image file was uploaded
#     if 'receipt' not in request.FILES:
#         return JsonResponse({'error': 'No receipt image provided', 'success': False}, status=400)
    
#     receipt_file = request.FILES['receipt']
    
#     # Get user and account
#     user = User.objects.get(username='demo_user')
#     account = user.accounts.first()
    
#     if not account:
#         return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
#     try:
#         # Parse receipt
#         service = get_receipt_service()
#         result = service.process_receipt(receipt_file)
        
#         if result['success']:
#             # Try to match to transaction
#             receipt_data = result['data']
#             if receipt_data.get('total'):
#                 match = service.match_receipt_to_transaction(receipt_data, account.id)
#                 result['matched_transaction'] = match
        
#         # Clean any Decimal values
#         result = clean_for_json_field(result)
        
#         return JsonResponse(result, encoder=CustomJSONEncoder)
        
#     except Exception as e:
#         print(f"Receipt parsing error: {e}")
#         import traceback
#         traceback.print_exc()
#         return JsonResponse({
#             'success': False,
#             'error': str(e),
#             'data': None
#         }, status=500)

# Replace parse_receipt_api in dashboard/views.py with this proper implementation

@csrf_exempt
@require_http_methods(["POST"])
def parse_receipt_api(request):
    """API endpoint for receipt parsing using TrOCR"""
    from dashboard.ml_services.receipt_service import get_receipt_service
    
    print("Receipt parsing request received")
    
    # Check if image file was uploaded
    if 'receipt' not in request.FILES:
        return JsonResponse({'error': 'No receipt image provided', 'success': False}, status=400)
    
    receipt_file = request.FILES['receipt']
    print(f"Processing file: {receipt_file.name}, size: {receipt_file.size} bytes")
    
    # Get user and account
    user = User.objects.get(username='demo_user')
    account = user.accounts.first()
    
    if not account:
        return JsonResponse({'error': 'No account found', 'success': False}, status=404)
    
    try:
        # Get the receipt service
        print("Loading receipt service...")
        service = get_receipt_service()
        
        # Process the receipt
        print("Processing receipt with TrOCR...")
        result = service.process_receipt(receipt_file, enhance=True)
        
        print(f"Processing result: success={result.get('success')}")
        
        if result['success']:
            # Try to match to transaction
            receipt_data = result['data']
            print(f"Extracted data: merchant={receipt_data.get('merchant')}, total={receipt_data.get('total')}")
            
            if receipt_data.get('total'):
                print(f"Attempting to match transaction for amount ${receipt_data['total']}")
                match = service.match_receipt_to_transaction(receipt_data, account.id)
                if match:
                    print(f"Found matching transaction: {match.get('description')}")
                    result['matched_transaction'] = match
                else:
                    print("No matching transaction found")
        else:
            print(f"Receipt parsing failed: {result.get('error')}")
        
        # Clean any Decimal values for JSON serialization
        result = clean_for_json_field(result)
        
        return JsonResponse(result, encoder=CustomJSONEncoder)
        
    except Exception as e:
        print(f"Receipt parsing error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e),
            'data': None
        }, status=500)

@require_http_methods(["GET"])
def generate_sample_receipt(request):
    """Generate a sample receipt for testing"""
    from PIL import Image, ImageDraw, ImageFont
    from django.http import HttpResponse
    import io
    
    receipt_type = request.GET.get('type', 'grocery')
    
    # Create a simple receipt image
    width, height = 400, 600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a better font, fallback to default
    try:
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    except:
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Receipt content based on type
    receipts = {
        'grocery': {
            'merchant': 'WHOLE FOODS',
            'items': [
                ('Organic Bananas', 3.99),
                ('Almond Milk', 4.49),
                ('Whole Wheat Bread', 3.29),
                ('Free Range Eggs', 5.99),
                ('Avocados (3)', 4.50),
            ],
            'subtotal': 22.26,
            'tax': 1.95,
            'total': 24.21,
            'date': '09/21/2025',
            'time': '14:32'
        },
        'coffee': {
            'merchant': 'STARBUCKS',
            'items': [
                ('Venti Latte', 5.95),
                ('Blueberry Muffin', 3.50),
                ('Extra Shot', 0.75),
            ],
            'subtotal': 10.20,
            'tax': 0.89,
            'total': 11.09,
            'date': '09/21/2025',
            'time': '08:45'
        },
        'restaurant': {
            'merchant': 'BLUE MOON CAFE',
            'items': [
                ('Caesar Salad', 12.50),
                ('Grilled Salmon', 28.95),
                ('House Wine', 8.50),
                ('Tiramisu', 9.95),
            ],
            'subtotal': 59.90,
            'tax': 5.24,
            'total': 65.14,
            'date': '09/20/2025',
            'time': '19:30'
        },
        'pharmacy': {
            'merchant': 'WALGREENS',
            'items': [
                ('Vitamin D3', 12.99),
                ('Ibuprofen 200mg', 8.49),
                ('Band-Aids', 5.99),
            ],
            'subtotal': 27.47,
            'tax': 2.40,
            'total': 29.87,
            'date': '09/21/2025',
            'time': '11:15'
        }
    }
    
    receipt = receipts.get(receipt_type, receipts['grocery'])
    
    # Draw receipt
    y = 20
    
    # Merchant name
    draw.text((width//2 - len(receipt['merchant'])*5, y), receipt['merchant'], fill='black', font=font_large)
    y += 40
    
    # Date and time
    draw.text((50, y), f"Date: {receipt['date']}", fill='black', font=font_normal)
    draw.text((250, y), f"Time: {receipt['time']}", fill='black', font=font_normal)
    y += 30
    
    draw.line([(30, y), (width-30, y)], fill='black', width=1)
    y += 20
    
    # Items
    for item_name, price in receipt['items']:
        draw.text((50, y), item_name, fill='black', font=font_normal)
        draw.text((width - 80, y), f"${price:.2f}", fill='black', font=font_normal)
        y += 25
    
    y += 10
    draw.line([(30, y), (width-30, y)], fill='black', width=1)
    y += 20
    
    # Totals
    draw.text((50, y), "Subtotal:", fill='black', font=font_normal)
    draw.text((width - 80, y), f"${receipt['subtotal']:.2f}", fill='black', font=font_normal)
    y += 25
    
    draw.text((50, y), "Tax:", fill='black', font=font_normal)
    draw.text((width - 80, y), f"${receipt['tax']:.2f}", fill='black', font=font_normal)
    y += 25
    
    draw.text((50, y), "TOTAL:", fill='black', font=font_large)
    draw.text((width - 90, y), f"${receipt['total']:.2f}", fill='black', font=font_large)
    y += 40
    
    draw.line([(30, y), (width-30, y)], fill='black', width=1)
    y += 20
    
    draw.text((width//2 - 60, y), "THANK YOU!", fill='black', font=font_normal)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    response = HttpResponse(img_bytes, content_type='image/png')
    response['Content-Disposition'] = f'inline; filename="sample_{receipt_type}.png"'
    return response


