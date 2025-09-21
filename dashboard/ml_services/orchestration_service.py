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
    
    def _extract_tools_from_response(self, claude_text: str, user_query: str) -> List[str]:
        """
        Extract which tools to run based on Claude's analysis or keywords
        """
        tools = []
        query_lower = user_query.lower()
        
        # Check for categorization intent
        if any(word in query_lower for word in ['categorize', 'category', 'categories', 'spending breakdown']):
            tools.append('categorize')
        
        # Check for forecasting intent
        if any(word in query_lower for word in ['forecast', 'predict', 'future', 'will i', 'next week', 'next month', 'balance in']):
            tools.append('forecast')
        
        # Check for fraud/anomaly intent
        if any(word in query_lower for word in ['fraud', 'suspicious', 'unusual', 'anomaly', 'weird', 'strange']):
            tools.append('fraud')
        
        # Check for summary/overview intent
        if any(word in query_lower for word in ['summary', 'overview', 'spending', 'how much', 'analysis', 'insights']):
            tools.append('summary')
        
        # Default to overview if no specific intent
        if not tools:
            tools.append('overview')
        
        return tools
    
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
                service = get_categorization_service()
                stats = service.get_category_statistics(account_id)
                results['categorization'] = stats
            except Exception as e:
                logger.error(f"Categorization failed: {e}")
        
        if 'forecast' in tools:
            try:
                service = get_forecasting_service()
                forecast = service.forecast_balance(account_id, horizon=7)
                results['forecast'] = forecast
            except Exception as e:
                logger.error(f"Forecasting failed: {e}")
        
        if 'fraud' in tools:
            try:
                service = get_fraud_service()
                # Check if account has Kaggle data or Plaid data
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
                logger.error(f"Fraud detection failed: {e}")
        
        if 'summary' in tools or 'overview' in tools:
            try:
                summary = self._get_account_summary(account_id)
                results['summary'] = summary
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
        
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
            if cat_data.get('categories'):
                insights.append(f"✅ Analyzed {cat_data['total_transactions']} categorized transactions")
                
                # Prepare chart data
                charts.append({
                    'type': 'pie',
                    'title': 'Spending by Category',
                    'data': [
                        {
                            'category': cat['ai_category'],
                            'amount': abs(float(cat['total_amount'] or 0))
                        }
                        for cat in cat_data['categories'][:8]
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
            
            # Prepare forecast chart
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
            
            if 'metrics' in fraud_data:  # Kaggle XGBoost results
                insights.append(
                    f"🔍 Fraud detection: {fraud_data['predicted_fraud_count']} suspicious transactions found"
                )
                if fraud_data.get('top_risks'):
                    insights.append("🚨 High-risk transactions detected - review needed")
            elif 'anomalies' in fraud_data:  # Plaid anomaly results
                if fraud_data['anomalies_found'] > 0:
                    insights.append(
                        f"🔍 Found {fraud_data['anomalies_found']} unusual transactions"
                    )
        
        # Process summary results
        if 'summary' in results:
            summary = results['summary']
            insights.append(
                f"📊 Current balance: ${summary['current_balance']:,.2f}"
            )
            insights.append(
                f"💳 Last 30 days: ${summary['last_30_days']['total_spent']:,.2f} spent"
            )
        
        # Build response
        response = {
            'success': True,
            'query': user_query,
            'timestamp': datetime.now().isoformat(),
            'insights': insights,
            'data': results,
            'charts': charts,
            'natural_language_response': self._create_natural_response(
                user_query, results, insights
            ),
            'suggested_actions': self._suggest_actions(results)
        }
        
        return response
    
    def _create_natural_response(self, query: str, results: Dict, insights: List[str]) -> str:
        """
        Create a natural language response
        """
        
        # Simple template-based response for now
        response_parts = []
        
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
        
        if 'categorization' in results:
            cat_data = results['categorization']
            if cat_data.get('categories'):
                top_cat = cat_data['categories'][0]
                response_parts.append(
                    f"Your highest spending category is {top_cat['ai_category'].replace('_', ' ').title()} "
                    f"with ${abs(top_cat['total_amount']):,.2f} spent. "
                )
        
        if 'fraud' in results:
            if results['fraud'].get('anomalies_found', 0) > 0:
                response_parts.append(
                    "I've detected some unusual transactions that you might want to review. "
                )
        
        return ''.join(response_parts) if response_parts else "I've analyzed your account data. Here are the key insights."
    
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