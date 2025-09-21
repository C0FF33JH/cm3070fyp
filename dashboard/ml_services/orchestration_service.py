"""
Claude Orchestration Service - The AI Brain
Interprets user queries and coordinates ML models
File: dashboard/ml_services/orchestration_service.py
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
from anthropic import Anthropic
import re

logger = logging.getLogger(__name__)

class ClaudeOrchestrationService:
    """
    Orchestrates ML models through Claude AI
    Acts as the intelligent coordinator for all banking insights
    """
    
    def __init__(self):
        self.client = None
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-3-haiku-20240307"  # Fast and cost-effective
        self.initialize_client()
        
        # Tool definitions for Claude
        self.tools = [
            {
                "name": "categorize_transactions",
                "description": "Categorize transactions using SetFit model",
                "parameters": {
                    "account_id": "ID of account to categorize transactions for"
                }
            },
            {
                "name": "forecast_cashflow",
                "description": "Forecast future cash flow and account balance",
                "parameters": {
                    "account_id": "ID of account to forecast",
                    "days": "Number of days to forecast (default 7)",
                    "include_insights": "Whether to include spending insights"
                }
            },
            {
                "name": "detect_fraud",
                "description": "Detect fraudulent or anomalous transactions",
                "parameters": {
                    "account_id": "ID of account to check",
                    "data_type": "Type of data: 'kaggle' for fraud detection or 'plaid' for anomalies"
                }
            },
            {
                "name": "get_spending_summary",
                "description": "Get spending summary and statistics by category",
                "parameters": {
                    "account_id": "ID of account",
                    "days": "Number of days to analyze (default 30)"
                }
            },
            {
                "name": "get_account_overview",
                "description": "Get basic account information and recent transactions",
                "parameters": {
                    "account_id": "ID of account"
                }
            }
        ]
        
    def initialize_client(self):
        """Initialize Anthropic client"""
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Using mock mode.")
            self.client = None
        else:
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Claude client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                self.client = None
    
    def process_query(self, user_query: str, account_id: int, context: Dict = None) -> Dict:
        """
        Process user query and orchestrate appropriate ML models
        
        Args:
            user_query: Natural language query from user
            account_id: User's account ID
            context: Additional context (previous messages, etc.)
            
        Returns:
            Response with insights and data
        """
        
        # If no API key, use rule-based routing
        if not self.client:
            return self._rule_based_routing(user_query, account_id)
        
        # Use Claude to understand intent and orchestrate
        try:
            response = self._claude_orchestration(user_query, account_id, context)
            return response
        except Exception as e:
            logger.error(f"Claude orchestration failed: {e}")
            return self._rule_based_routing(user_query, account_id)
    
    def _claude_orchestration(self, user_query: str, account_id: int, context: Dict = None) -> Dict:
        """
        Use Claude to understand query and orchestrate models
        """
        
        # Build system prompt
        system_prompt = """You are an AI banking assistant with access to specialized ML models.
        
Your capabilities:
1. Transaction Categorization (SetFit) - Categorize transactions into spending categories
2. Cash Flow Forecasting (Chronos-T5) - Predict future account balances
3. Fraud Detection (XGBoost) - Identify suspicious transactions
4. Spending Analysis - Analyze spending patterns and provide insights

Based on the user's query, determine which tools to use and provide helpful insights.
Be concise but informative. Focus on actionable insights."""

        # Build the conversation
        messages = [
            {
                "role": "user",
                "content": f"""User Query: {user_query}
                
Account ID: {account_id}

Please analyze this query and determine which ML models/tools should be used.
Then provide a helpful response with insights."""
            }
        ]
        
        # Get Claude's response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=messages
        )
        
        # Parse Claude's response and execute tools
        claude_text = response.content[0].text
        tools_to_run = self._extract_tools_from_response(claude_text, user_query)
        
        # Execute the identified tools
        results = self._execute_tools(tools_to_run, account_id)
        
        # Generate final response with Claude
        final_response = self._generate_final_response(
            user_query, results, claude_text, account_id
        )
        
        return final_response
    
    # def _extract_tools_from_response(self, claude_text: str, user_query: str) -> List[str]:
    #     """
    #     Extract which tools to run based on Claude's analysis or keywords
    #     More specific routing logic
    #     """
    #     tools = []
    #     query_lower = user_query.lower()
        
    #     # Check for categorization intent - must be explicit
    #     if any(word in query_lower for word in ['categorize', 'category', 'categories', 'breakdown by category', 'spending by category']):
    #         tools.append('categorize')
        
    #     # Check for forecasting intent
    #     if any(word in query_lower for word in ['forecast', 'predict', 'future', 'will i', 'next week', 'next month', 'balance in', 'enough money']):
    #         tools.append('forecast')
        
    #     # Check for fraud/anomaly intent
    #     if any(word in query_lower for word in ['fraud', 'suspicious', 'unusual', 'anomaly', 'anomalous', 'weird', 'strange']):
    #         tools.append('fraud')
        
    #     # Check for spending analysis intent
    #     if any(phrase in query_lower for phrase in ['how much did i spend', 'spending on', 'spent on', 'spending pattern', 'analyze my spending']):
    #         tools.append('summary')
    #         # Also add categorization for spending questions
    #         if any(word in query_lower for word in ['food', 'transport', 'entertainment', 'medical']):
    #             tools.append('categorize')
        
    #     # Check for balance/overview intent
    #     if any(word in query_lower for word in ['balance', 'overview', 'summary']) and len(tools) == 0:
    #         tools.append('summary')
        
    #     # Default to overview if no specific intent
    #     if not tools:
    #         tools.append('overview')
        
    #     return tools


    # def _analyze_user_query(self, user_query: str) -> List[str]:
    #     """
    #     Analyze the user query and determine which tools to use
    #     """
    #     Execute the identified ML model tools
    #     """
    #     from dashboard.ml_services import (
    #         get_categorization_service,
    #         get_forecasting_service,
    #         get_fraud_service
    #     )
        
    #     results = {}
        
    #     if 'categorize' in tools:
    #         try:
    #             service = get_categorization_service()
    #             stats = service.get_category_statistics(account_id)
    #             results['categorization'] = stats
    #         except Exception as e:
    #             logger.error(f"Categorization failed: {e}")
        
    #     if 'forecast' in tools:
    #         try:
    #             service = get_forecasting_service()
    #             forecast = service.forecast_balance(account_id, horizon=7)
    #             results['forecast'] = forecast
    #         except Exception as e:
    #             logger.error(f"Forecasting failed: {e}")
        
    #     if 'fraud' in tools:
    #         try:
    #             service = get_fraud_service()
    #             # Check if account has Kaggle data or Plaid data
    #             from dashboard.models import Transaction
    #             has_kaggle = Transaction.objects.filter(
    #                 account_id=account_id,
    #                 data_source='kaggle'
    #             ).exists()
                
    #             if has_kaggle:
    #                 fraud_results = service.analyze_batch(account_id)
    #             else:
    #                 fraud_results = service.detect_anomalies_plaid(account_id)
    #             results['fraud'] = fraud_results
    #         except Exception as e:
    #             logger.error(f"Fraud detection failed: {e}")
        
    #     if 'summary' in tools or 'overview' in tools:
    #         try:
    #             summary = self._get_account_summary(account_id)
    #             results['summary'] = summary
    #         except Exception as e:
    #             logger.error(f"Summary generation failed: {e}")
        
    #     return results

    # Replace the _extract_tools_from_response method in orchestration_service.py


# Add this method to the ClaudeOrchestrationService class in orchestration_service.py

    def _get_account_summary(self, account_id: int) -> Dict:
        """
        Get account summary and recent activity
        """
        from dashboard.models import Account, Transaction
        from django.db.models import Sum, Count, Avg
        from django.utils import timezone
        from datetime import timedelta
        
        try:
            account = Account.objects.get(id=account_id)
            
            # Recent transactions
            recent_trans = Transaction.objects.filter(
                account_id=account_id,
                data_source='plaid'
            ).order_by('-date')[:10]
            
            # Last 30 days stats
            thirty_days_ago = timezone.now() - timedelta(days=30)
            stats = Transaction.objects.filter(
                account_id=account_id,
                date__gte=thirty_days_ago,
                data_source='plaid'
            ).aggregate(
                total_spent=Sum('amount'),
                num_transactions=Count('id'),
                avg_transaction=Avg('amount')
            )
            
            # Category breakdown
            categories = Transaction.objects.filter(
                account_id=account_id,
                date__gte=thirty_days_ago,
                ai_category__isnull=False,
                amount__lt=0
            ).values('ai_category').annotate(
                total=Sum('amount'),
                count=Count('id')
            ).order_by('total')[:5]
            
            return {
                'account_name': account.account_name,
                'current_balance': float(account.balance),
                'last_30_days': {
                    'total_spent': abs(float(stats['total_spent'] or 0)),
                    'num_transactions': stats['num_transactions'],
                    'avg_transaction': abs(float(stats['avg_transaction'] or 0))
                },
                'top_categories': [
                    {
                        'category': cat['ai_category'],
                        'amount': abs(float(cat['total'])),
                        'count': cat['count']
                    }
                    for cat in categories
                ],
                'recent_transactions': [
                    {
                        'date': trans.date.isoformat(),
                        'description': trans.description,
                        'amount': float(trans.amount),
                        'category': trans.ai_category
                    }
                    for trans in recent_trans[:5]
                ]
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {'error': str(e)}

    def _extract_tools_from_response(self, claude_text: str, user_query: str) -> List[str]:
        """
        Extract which tools to run based on Claude's analysis or keywords
        More specific routing logic
        """
        tools = []
        query_lower = user_query.lower()
        
        # Check for categorization intent - be more flexible with terms
        categorize_keywords = ['categorize', 'categorise', 'category', 'categories', 
                            'breakdown', 'spending by category', 'organize', 'classification']
        if any(word in query_lower for word in categorize_keywords):
            tools.append('categorize')
            # Also get summary for context
            tools.append('summary')
        
        # Check for forecasting intent
        forecast_keywords = ['forecast', 'predict', 'future', 'will i', 'next week', 
                            'next month', 'balance in', 'enough money', 'cash flow']
        if any(word in query_lower for word in forecast_keywords):
            tools.append('forecast')
        
        # Check for fraud/anomaly intent
        fraud_keywords = ['fraud', 'suspicious', 'unusual', 'anomaly', 'anomalous', 
                        'weird', 'strange', 'unauthorized', 'fraudulent']
        if any(word in query_lower for word in fraud_keywords):
            tools.append('fraud')
        
        # Check for spending analysis intent
        spending_keywords = ['how much did i spend', 'spending on', 'spent on', 
                            'spending pattern', 'analyze my spending', 'expenses']
        if any(phrase in query_lower for phrase in spending_keywords):
            tools.append('summary')
            # Also add categorization for spending questions
            if not 'categorize' in tools:
                tools.append('categorize')
        
        # Check for balance/overview intent
        if any(word in query_lower for word in ['balance', 'overview', 'summary']):
            if not tools:  # Only if no other tools identified
                tools.append('summary')
        
        # Default to overview if no specific intent
        if not tools:
            tools.append('overview')
        
        # Debug logging
        print(f"Query: {user_query}")
        print(f"Identified tools: {tools}")
        
        return tools

    # Also update the _execute_tools method to ensure categorization works

    def _execute_tools(self, tools: List[str], account_id: int) -> Dict:
        """
        Execute the identified ML model tools
        """
        from dashboard.ml_services import (
            get_categorization_service,
            get_forecasting_service,
            get_fraud_service
        )
        
        results = {}
        
        if 'categorize' in tools:
            try:
                print(f"Executing categorization for account {account_id}")
                service = get_categorization_service()
                
                # First try to categorize any uncategorized transactions
                count = service.categorize_uncategorized_transactions(account_id)
                print(f"Categorized {count} new transactions")
                
                # Then get the statistics
                stats = service.get_category_statistics(account_id)
                print(f"Got statistics: {stats.get('total_transactions', 0)} transactions")
                
                results['categorization'] = {
                    'newly_categorized': count,
                    'statistics': stats,
                    'success': True
                }
            except Exception as e:
                print(f"Categorization failed: {e}")
                import traceback
                traceback.print_exc()
                results['categorization'] = {'error': str(e), 'success': False}
        
        if 'forecast' in tools:
            try:
                print(f"Executing forecast for account {account_id}")
                service = get_forecasting_service()
                forecast = service.forecast_balance(account_id, horizon=7)
                results['forecast'] = forecast
            except Exception as e:
                print(f"Forecasting failed: {e}")
                results['forecast'] = {'error': str(e), 'success': False}
        
        if 'fraud' in tools:
            try:
                print(f"Executing fraud detection for account {account_id}")
                service = get_fraud_service()
                from dashboard.models import Transaction
                has_kaggle = Transaction.objects.filter(
                    account_id=account_id,
                    data_source='kaggle'
                ).exists()
                
                if has_kaggle:
                    fraud_results = service.analyze_batch(account_id)
                else:
                    fraud_results = service.detect_anomalies_plaid(account_id)
                results['fraud'] = fraud_results
            except Exception as e:
                print(f"Fraud detection failed: {e}")
                results['fraud'] = {'error': str(e), 'success': False}
        
        if 'summary' in tools or 'overview' in tools:
            try:
                print(f"Getting account summary for account {account_id}")
                summary = self._get_account_summary(account_id)
                results['summary'] = summary
            except Exception as e:
                print(f"Summary generation failed: {e}")
                results['summary'] = {'error': str(e), 'success': False}
        
        print(f"Results keys: {results.keys()}")
        return results
        
        def _get_account_summary(self, account_id: int) -> Dict:
            """
            Get account summary and recent activity
            """
            from dashboard.models import Account, Transaction
            from django.db.models import Sum, Count, Avg
            
            account = Account.objects.get(id=account_id)
            
            # Recent transactions
            recent_trans = Transaction.objects.filter(
                account_id=account_id,
                data_source='plaid'
            ).order_by('-date')[:10]
            
            # Last 30 days stats
            thirty_days_ago = timezone.now() - timedelta(days=30)
            stats = Transaction.objects.filter(
                account_id=account_id,
                date__gte=thirty_days_ago,
                data_source='plaid'
            ).aggregate(
                total_spent=Sum('amount'),
                num_transactions=Count('id'),
                avg_transaction=Avg('amount')
            )
            
            # Category breakdown
            categories = Transaction.objects.filter(
                account_id=account_id,
                date__gte=thirty_days_ago,
                ai_category__isnull=False,
                amount__lt=0
            ).values('ai_category').annotate(
                total=Sum('amount'),
                count=Count('id')
            ).order_by('total')[:5]
            
            return {
                'account_name': account.account_name,
                'current_balance': float(account.balance),
                'last_30_days': {
                    'total_spent': abs(float(stats['total_spent'] or 0)),
                    'num_transactions': stats['num_transactions'],
                    'avg_transaction': abs(float(stats['avg_transaction'] or 0))
                },
                'top_categories': [
                    {
                        'category': cat['ai_category'],
                        'amount': abs(float(cat['total'])),
                        'count': cat['count']
                    }
                    for cat in categories
                ],
                'recent_transactions': [
                    {
                        'date': trans.date.isoformat(),
                        'description': trans.description,
                        'amount': float(trans.amount),
                        'category': trans.ai_category
                    }
                    for trans in recent_trans[:5]
                ]
            }
        
    # def _generate_final_response(
    #     self, 
    #     user_query: str, 
    #     results: Dict, 
    #     claude_analysis: str,
    #     account_id: int
    # ) -> Dict:
    #     """
    #     Generate final response combining all results
    #     """
        
    #     # Build insights from results
    #     insights = []
    #     charts = []
        
    #     # Process categorization results
    #     if 'categorization' in results:
    #         cat_data = results['categorization']
    #         if cat_data.get('categories'):
    #             insights.append(f"✅ Analyzed {cat_data['total_transactions']} categorized transactions")
                
    #             # Prepare chart data
    #             charts.append({
    #                 'type': 'pie',
    #                 'title': 'Spending by Category',
    #                 'data': [
    #                     {
    #                         'category': cat['ai_category'],
    #                         'amount': abs(float(cat['total_amount'] or 0))
    #                     }
    #                     for cat in cat_data['categories'][:8]
    #                 ]
    #             })
        
    #     # Process forecast results
    #     if 'forecast' in results and results['forecast'].get('success'):
    #         forecast = results['forecast']
    #         fore_insights = forecast.get('insights', {})
            
    #         insights.append(
    #             f"📈 7-day forecast: Balance will {fore_insights.get('trend', 'remain stable')}"
    #         )
    #         insights.append(
    #             f"💰 Predicted balance: ${fore_insights.get('predicted_balance_7d', 0):,.2f}"
    #         )
            
    #         if 'warnings' in fore_insights:
    #             for warning in fore_insights['warnings']:
    #                 insights.append(f"⚠️ {warning}")
            
    #         # Prepare forecast chart
    #         charts.append({
    #             'type': 'line',
    #             'title': 'Cash Flow Forecast',
    #             'data': {
    #                 'historical': forecast.get('historical', {}),
    #                 'forecast': forecast.get('forecast', {})
    #             }
    #         })
        
    #     # Process fraud results
    #     if 'fraud' in results:
    #         fraud_data = results['fraud']
            
    #         if 'metrics' in fraud_data:  # Kaggle XGBoost results
    #             insights.append(
    #                 f"🔍 Fraud detection: {fraud_data['predicted_fraud_count']} suspicious transactions found"
    #             )
    #             if fraud_data.get('top_risks'):
    #                 insights.append("🚨 High-risk transactions detected - review needed")
    #         elif 'anomalies' in fraud_data:  # Plaid anomaly results
    #             if fraud_data['anomalies_found'] > 0:
    #                 insights.append(
    #                     f"🔍 Found {fraud_data['anomalies_found']} unusual transactions"
    #                 )
        
    #     # Process summary results
    #     if 'summary' in results:
    #         summary = results['summary']
    #         insights.append(
    #             f"📊 Current balance: ${summary['current_balance']:,.2f}"
    #         )
    #         insights.append(
    #             f"💳 Last 30 days: ${summary['last_30_days']['total_spent']:,.2f} spent"
    #         )
            
    #         # Add category-specific spending if asked about specific category
    #         query_lower = user_query.lower()
    #         for cat in summary.get('top_categories', []):
    #             cat_name = cat['category'].lower().replace('_', ' ')
    #             if cat_name in query_lower or cat['category'].lower() in query_lower:
    #                 insights.append(
    #                     f"🍔 {cat['category'].replace('_', ' ').title()}: ${cat['amount']:,.2f} ({cat['count']} transactions)"
    #                 )
        
    #     # Build response
    #     response = {
    #         'success': True,
    #         'query': user_query,
    #         'timestamp': datetime.now().isoformat(),
    #         'insights': insights,
    #         'data': results,
    #         'charts': charts,
    #         'natural_language_response': self._create_natural_response(
    #             user_query, results, insights
    #         ),
    #         'suggested_actions': self._suggest_actions(results)
    #     }
        
    #     return response
    
    # def _create_natural_response(self, query: str, results: Dict, insights: List[str]) -> str:
    #     """
    #     Create a natural language response
    #     """
        
    #     # Simple template-based response for now
    #     response_parts = []
        
    #     if 'summary' in results:
    #         summary = results['summary']
    #         response_parts.append(
    #             f"Your {summary['account_name']} has a current balance of ${summary['current_balance']:,.2f}. "
    #         )
        
    #     if 'forecast' in results and results['forecast'].get('success'):
    #         forecast = results['forecast']['insights']
    #         response_parts.append(
    #             f"Based on your spending patterns, your balance is expected to be "
    #             f"${forecast['predicted_balance_7d']:,.2f} in 7 days. "
    #         )
        
    #     if 'categorization' in results:
    #         cat_data = results['categorization']
    #         if cat_data.get('categories'):
    #             # Find actual spending categories (negative amounts)
    #             spending_cats = [c for c in cat_data['categories'] if c.get('total_amount', 0) < 0]
    #             if spending_cats:
    #                 top_cat = spending_cats[0]
    #                 response_parts.append(
    #                     f"Your highest spending category is {top_cat['ai_category'].replace('_', ' ').title()} "
    #                     f"with ${abs(top_cat['total_amount']):,.2f} spent. "
    #                 )
        
    #     if 'fraud' in results:
    #         if results['fraud'].get('anomalies_found', 0) > 0:
    #             response_parts.append(
    #                 "I've detected some unusual transactions that you might want to review. "
    #             )
        
    #     return ''.join(response_parts) if response_parts else "I've analyzed your account data. Here are the key insights."

    # Update the _create_natural_response method in orchestration_service.py

    def _create_natural_response(self, query: str, results: Dict, insights: List[str]) -> str:
        """
        Create a natural language response
        """
        
        response_parts = []
        
        # Handle categorization requests specifically
        if 'categoriz' in query.lower() and 'categorization' in results:
            cat_data = results['categorization']
            if cat_data.get('success'):
                stats = cat_data.get('statistics', {})
                
                # Check if new categorization was done
                newly_categorized = cat_data.get('newly_categorized', 0)
                total_transactions = stats.get('total_transactions', 0)
                
                if newly_categorized > 0:
                    response_parts.append(
                        f"I've successfully categorized {newly_categorized} transactions! "
                    )
                else:
                    response_parts.append(
                        f"All {total_transactions} transactions are already categorized. "
                    )
                
                # Get spending categories (negative amounts only)
                categories = stats.get('categories', [])
                spending_cats = [c for c in categories if c.get('total_amount', 0) < 0]
                
                if spending_cats:
                    # Sort by amount (most spent first)
                    spending_cats.sort(key=lambda x: x.get('total_amount', 0))
                    
                    # Top spending categories summary
                    response_parts.append(f"Here's your spending breakdown: ")
                    
                    # Show top 5 categories
                    for i, cat in enumerate(spending_cats[:5], 1):
                        cat_name = cat['ai_category'].replace('_', ' ').title()
                        amount = abs(float(cat.get('total_amount', 0)))
                        count = cat.get('count', 0)
                        response_parts.append(
                            f"\n{i}. {cat_name}: ${amount:,.2f} ({count} transactions)"
                        )
                    
                    # Add total spending
                    total_spent = sum(abs(float(c.get('total_amount', 0))) for c in spending_cats)
                    response_parts.append(f"\n\nTotal spending: ${total_spent:,.2f}")
                    
                    # Add confidence if available
                    avg_conf = stats.get('average_confidence', 0)
                    if avg_conf:
                        response_parts.append(f" with {avg_conf*100:.1f}% average confidence.")
                
                return ''.join(response_parts)
        
        # Handle other queries (existing code)
        if 'summary' in results:
            summary = results['summary']
            response_parts.append(
                f"Your {summary['account_name']} has a current balance of ${summary['current_balance']:,.2f}. "
            )
        
        if 'forecast' in results and results['forecast'].get('success'):
            forecast = results['forecast']['insights']
            response_parts.append(
                f"Based on your spending patterns, your balance is expected to be "
                f"${forecast['predicted_balance_7d']:,.2f} in 7 days. "
            )
        
        if 'fraud' in results:
            if results['fraud'].get('anomalies_found', 0) > 0:
                response_parts.append(
                    "I've detected some unusual transactions that you might want to review. "
                )
        
        return ''.join(response_parts) if response_parts else "I've analyzed your account data. Here are the key insights."

    # Also fix the chart generation in _generate_final_response

    # def _generate_final_response(
    #     self, 
    #     user_query: str, 
    #     results: Dict, 
    #     claude_analysis: str,
    #     account_id: int
    # ) -> Dict:
    #     """
    #     Generate final response combining all results
    #     """
        
    #     # Build insights from results
    #     insights = []
    #     charts = []
        
    #     # Process categorization results
    #     if 'categorization' in results:
    #         cat_data = results['categorization']
    #         if cat_data.get('success'):
    #             stats = cat_data.get('statistics', {})
                
    #             # Add insights
    #             if cat_data.get('newly_categorized', 0) > 0:
    #                 insights.append(f"✅ Categorized {cat_data['newly_categorized']} new transactions")
                
    #             if stats.get('total_transactions'):
    #                 insights.append(f"📊 Total transactions: {stats['total_transactions']}")
                
    #             if stats.get('average_confidence'):
    #                 insights.append(f"🎯 Average confidence: {(stats['average_confidence'] * 100):.1f}%")
                
    #             # Create proper pie chart with ALL spending categories
    #             categories = stats.get('categories', [])
    #             spending_cats = [c for c in categories if c.get('total_amount', 0) < 0]
                
    #             if spending_cats:
    #                 # Sort by amount (highest spending first)
    #                 spending_cats.sort(key=lambda x: x.get('total_amount', 0))
                    
    #                 # Add top spending insight
    #                 top_cat = spending_cats[0]
    #                 insights.append(
    #                     f"💰 Highest spending: {top_cat['ai_category'].replace('_', ' ').title()} "
    #                     f"(${abs(float(top_cat.get('total_amount', 0))):,.2f})"
    #                 )
                    
    #                 # Create chart with all categories (or top 8 for visibility)
    #                 charts.append({
    #                     'type': 'pie',
    #                     'title': 'Spending by Category',
    #                     'data': [
    #                         {
    #                             'category': cat['ai_category'].replace('_', ' ').title(),
    #                             'amount': abs(float(cat.get('total_amount', 0)))
    #                         }
    #                         for cat in spending_cats[:8]  # Top 8 categories
    #                     ]
    #                 })
        
    #     # Process forecast results (keep existing)
    #     if 'forecast' in results and results['forecast'].get('success'):
    #         forecast = results['forecast']
    #         fore_insights = forecast.get('insights', {})
            
    #         insights.append(
    #             f"📈 7-day forecast: Balance will {fore_insights.get('trend', 'remain stable')}"
    #         )
    #         insights.append(
    #             f"💰 Predicted balance: ${fore_insights.get('predicted_balance_7d', 0):,.2f}"
    #         )
            
    #         if 'warnings' in fore_insights:
    #             for warning in fore_insights['warnings']:
    #                 insights.append(f"⚠️ {warning}")
            
    #         charts.append({
    #             'type': 'line',
    #             'title': 'Cash Flow Forecast',
    #             'data': {
    #                 'historical': forecast.get('historical', {}),
    #                 'forecast': forecast.get('forecast', {})
    #             }
    #         })
        
    #     # Process fraud results (keep existing)
    #     if 'fraud' in results:
    #         fraud_data = results['fraud']
            
    #         if 'metrics' in fraud_data:
    #             insights.append(
    #                 f"🔍 Fraud detection: {fraud_data['predicted_fraud_count']} suspicious transactions"
    #             )
    #         elif 'anomalies' in fraud_data:
    #             if fraud_data['anomalies_found'] > 0:
    #                 insights.append(
    #                     f"🔍 Found {fraud_data['anomalies_found']} unusual transactions"
    #                 )
        
    #     # Process summary results (keep existing)
    #     if 'summary' in results:
    #         summary = results['summary']
    #         insights.append(
    #             f"📊 Current balance: ${summary['current_balance']:,.2f}"
    #         )
    #         insights.append(
    #             f"💳 Last 30 days: ${summary['last_30_days']['total_spent']:,.2f} spent"
    #         )
        
    #     # Build natural language response
    #     response_text = self._create_natural_response(user_query, results, insights)
        
    #     # Build response
    #     response = {
    #         'success': True,
    #         'query': user_query,
    #         'timestamp': datetime.now().isoformat(),
    #         'insights': insights,
    #         'data': results,
    #         'charts': charts,
    #         'natural_language_response': response_text,
    #         'suggested_actions': self._suggest_actions(results)
    #     }
        
    #     return response
    def _generate_final_response(
            self, 
            user_query: str, 
            results: Dict, 
            claude_analysis: str,
            account_id: int
        ) -> Dict:
            """
            Generate final response combining all results
            """
            
            # Build insights from results
            insights = []
            charts = []
            
            # Process categorization results
            if 'categorization' in results:
                cat_data = results['categorization']
                if cat_data.get('success'):
                    stats = cat_data.get('statistics', {})
                    
                    # Add insights
                    if cat_data.get('newly_categorized', 0) > 0:
                        insights.append(f"✅ Categorized {cat_data['newly_categorized']} new transactions")
                    
                    if stats.get('total_transactions'):
                        insights.append(f"📊 Total transactions: {stats['total_transactions']}")
                    
                    if stats.get('average_confidence'):
                        insights.append(f"🎯 Average confidence: {(stats['average_confidence'] * 100):.1f}%")
                    
                    # Create proper pie chart with ALL spending categories
                    categories = stats.get('categories', [])
                    spending_cats = [c for c in categories if c.get('total_amount', 0) < 0]
                    
                    if spending_cats:
                        # Sort by amount (highest spending first)
                        spending_cats.sort(key=lambda x: x.get('total_amount', 0))
                        
                        # Add top spending insight
                        top_cat = spending_cats[0]
                        insights.append(
                            f"💰 Highest spending: {top_cat['ai_category'].replace('_', ' ').title()} "
                            f"(${abs(float(top_cat.get('total_amount', 0))):,.2f})"
                        )
                        
                        # Create chart with all categories (or top 8 for visibility)
                        charts.append({
                            'type': 'pie',
                            'title': 'Spending by Category',
                            'data': [
                                {
                                    'category': cat['ai_category'].replace('_', ' ').title(),
                                    'amount': abs(float(cat.get('total_amount', 0)))
                                }
                                for cat in spending_cats[:8]  # Top 8 categories
                            ]
                        })
            
            # Process forecast results
            if 'forecast' in results and results['forecast'].get('success'):
                forecast = results['forecast']
                fore_insights = forecast.get('insights', {})
                
                insights.append(
                    f"📈 7-day forecast: Balance will {fore_insights.get('trend', 'remain stable')}"
                )
                insights.append(
                    f"💰 Predicted balance: ${fore_insights.get('predicted_balance_7d', 0):,.2f}"
                )
                
                if 'warnings' in fore_insights:
                    for warning in fore_insights['warnings']:
                        insights.append(f"⚠️ {warning}")
                
                charts.append({
                    'type': 'line',
                    'title': 'Cash Flow Forecast',
                    'data': {
                        'historical': forecast.get('historical', {}),
                        'forecast': forecast.get('forecast', {})
                    }
                })
            
            # Process fraud results
            if 'fraud' in results:
                fraud_data = results['fraud']
                
                if 'metrics' in fraud_data:
                    insights.append(
                        f"🔍 Fraud detection: {fraud_data['predicted_fraud_count']} suspicious transactions"
                    )
                elif 'anomalies' in fraud_data:
                    if fraud_data['anomalies_found'] > 0:
                        insights.append(
                            f"🔍 Found {fraud_data['anomalies_found']} unusual transactions"
                        )
            
            # Process summary results - with error handling
            if 'summary' in results:
                summary = results['summary']
                # Check if summary has error or is missing current_balance
                if not summary.get('error') and 'current_balance' in summary:
                    insights.append(
                        f"📊 Current balance: ${summary['current_balance']:,.2f}"
                    )
                    if summary.get('last_30_days'):
                        insights.append(
                            f"💳 Last 30 days: ${summary['last_30_days']['total_spent']:,.2f} spent"
                        )
            
            # Build natural language response
            response_text = self._create_natural_response(user_query, results, insights)
            
            # Build response
            response = {
                'success': True,
                'query': user_query,
                'timestamp': datetime.now().isoformat(),
                'insights': insights,
                'data': results,
                'charts': charts,
                'natural_language_response': response_text,
                'suggested_actions': self._suggest_actions(results)
            }
            
            return response


    
    def _suggest_actions(self, results: Dict) -> List[str]:
        """
        Suggest actionable next steps based on analysis
        """
        actions = []
        
        # Based on forecast
        if 'forecast' in results and results['forecast'].get('success'):
            forecast = results['forecast']['insights']
            if forecast.get('risk_level') == 'high':
                actions.append("Review upcoming expenses to avoid negative balance")
            if forecast.get('trend') == 'decreasing':
                actions.append("Consider reducing discretionary spending")
        
        # Based on categorization
        if 'categorization' in results:
            cat_data = results['categorization']
            if cat_data.get('average_confidence', 0) < 0.8:
                actions.append("Review and correct transaction categories for better insights")
        
        # Based on fraud detection
        if 'fraud' in results:
            if results['fraud'].get('anomalies_found', 0) > 0:
                actions.append("Review flagged transactions for potential fraud")
        
        # General suggestions
        if 'summary' in results:
            summary = results['summary']
            if summary['last_30_days']['total_spent'] > summary['current_balance'] * 0.5:
                actions.append("Your spending rate is high - consider creating a budget")
        
        return actions[:3]  # Limit to top 3 suggestions
    
    def _rule_based_routing(self, user_query: str, account_id: int) -> Dict:
        """
        Fallback rule-based routing when Claude is not available
        """
        query_lower = user_query.lower()
        
        # Determine intent
        tools = self._extract_tools_from_response("", user_query)
        
        # Execute tools
        results = self._execute_tools(tools, account_id)
        
        # Generate response
        response = self._generate_final_response(
            user_query, results, "", account_id
        )
        
        return response


# Singleton instance
_orchestration_service = None

def get_orchestration_service() -> ClaudeOrchestrationService:
    """Get or create the singleton orchestration service"""
    global _orchestration_service
    if _orchestration_service is None:
        _orchestration_service = ClaudeOrchestrationService()
    return _orchestration_service