"""
Chronos-T5 Cash Flow Forecasting Service
Handles time series forecasting for account balances
File: dashboard/ml_services/forecasting_service.py
"""

import os
import logging
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from chronos import ChronosPipeline
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

class ChronosForecastingService:
    """Service for forecasting cash flow using Chronos-T5 model"""
    
    def __init__(self):
        self.pipeline = None
        self.config_path = os.path.join(settings.MODEL_DIR, 'chronos_config.json')
        self.model_id = "amazon/chronos-t5-small"  # Can upgrade to medium/large
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self) -> bool:
        """Load the Chronos-T5 model"""
        try:
            if self.pipeline is None:
                logger.info(f"Loading Chronos-T5 model: {self.model_id}")
                
                # Load configuration if exists
                if os.path.exists(self.config_path):
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                        self.model_id = config.get('model_id', self.model_id)
                
                # Initialize Chronos pipeline
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=torch.float32
                )
                
                logger.info(f"Chronos model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            return False
    
    def prepare_balance_series(self, account_id: int, days: int = 90) -> pd.Series:
        """
        Prepare daily balance time series from transactions
        
        Args:
            account_id: Account to get balances for
            days: Number of days of history to use
            
        Returns:
            Pandas Series with daily closing balances
        """
        from dashboard.models import Transaction, Account
        
        # Get account
        account = Account.objects.get(id=account_id)
        
        # Get transactions for the period
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        transactions = Transaction.objects.filter(
            account=account,
            date__gte=start_date,
            date__lte=end_date,
            data_source='plaid'  # Only use open banking data
        ).order_by('date').values('date', 'amount')
        
        if not transactions:
            logger.warning(f"No transactions found for account {account_id}")
            return pd.Series()
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate running balance
        # Start from current balance and work backwards
        current_balance = float(account.balance)
        
        # Group by date and sum amounts
        daily_changes = df.groupby(df['date'].dt.date)['amount'].sum()
        
        # Create complete date range
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        daily_balances = pd.Series(index=date_range, dtype=float)
        
        # Calculate daily balances
        # Work backwards from current balance
        for date in reversed(date_range):
            if date.date() == end_date.date():
                daily_balances[date] = current_balance
            else:
                next_date = date + timedelta(days=1)
                change = daily_changes.get(next_date.date(), 0)
                daily_balances[date] = daily_balances[next_date] - float(change)
        
        logger.info(f"Prepared {len(daily_balances)} days of balance data")
        return daily_balances
    
    def forecast_balance(
        self, 
        account_id: int, 
        horizon: int = 7,
        history_days: int = 90,
        num_samples: int = 20
    ) -> Dict:
        """
        Forecast future account balances
        
        Args:
            account_id: Account to forecast
            horizon: Days to forecast ahead
            history_days: Days of history to use
            num_samples: Number of forecast samples for uncertainty
            
        Returns:
            Dictionary with forecast results
        """
        if not self.load_model():
            return {'error': 'Model not loaded'}
        
        try:
            # Get historical balance series
            balance_series = self.prepare_balance_series(account_id, history_days)
            
            if len(balance_series) < 7:
                return {'error': 'Insufficient historical data for forecasting'}
            
            # Convert to tensor
            context = torch.tensor(balance_series.values, dtype=torch.float32)
            
            # Generate forecast
            logger.info(f"Generating {horizon}-day forecast with {num_samples} samples...")
            forecast = self.pipeline.predict(
                context=context.unsqueeze(0),  # Add batch dimension
                prediction_length=horizon,
                num_samples=num_samples
            )
            
            # Process forecast results
            forecast_values = forecast.numpy().squeeze()  # Remove batch dimension
            
            # Calculate statistics
            forecast_mean = np.mean(forecast_values, axis=0)
            forecast_median = np.median(forecast_values, axis=0)
            forecast_std = np.std(forecast_values, axis=0)
            forecast_lower = np.percentile(forecast_values, 10, axis=0)
            forecast_upper = np.percentile(forecast_values, 90, axis=0)
            
            # Create forecast dates
            last_date = balance_series.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Prepare result
            result = {
                'success': True,
                'account_id': account_id,
                'forecast_start': forecast_dates[0].isoformat(),
                'forecast_end': forecast_dates[-1].isoformat(),
                'horizon_days': horizon,
                'history_days': history_days,
                'num_samples': num_samples,
                'historical': {
                    'dates': [d.isoformat() for d in balance_series.index[-30:]],  # Last 30 days
                    'balances': balance_series.values[-30:].tolist()
                },
                'forecast': {
                    'dates': [d.isoformat() for d in forecast_dates],
                    'mean': forecast_mean.tolist(),
                    'median': forecast_median.tolist(),
                    'lower_bound': forecast_lower.tolist(),
                    'upper_bound': forecast_upper.tolist(),
                    'std_dev': forecast_std.tolist()
                },
                'insights': self._generate_insights(
                    balance_series,
                    forecast_mean,
                    forecast_std
                )
            }
            
            logger.info(f"Forecast completed successfully for {horizon} days")
            return result
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return {'error': str(e)}
    
    def _generate_insights(
        self, 
        historical: pd.Series, 
        forecast_mean: np.ndarray,
        forecast_std: np.ndarray
    ) -> Dict:
        """Generate insights from forecast results"""
        
        current_balance = historical.iloc[-1]
        predicted_balance = forecast_mean[-1]
        balance_change = predicted_balance - current_balance
        
        # Analyze trend
        if len(forecast_mean) > 1:
            trend = 'increasing' if forecast_mean[-1] > forecast_mean[0] else 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate volatility
        volatility = np.mean(forecast_std)
        volatility_level = 'high' if volatility > 0.2 * abs(current_balance) else 'low'
        
        # Risk assessment
        min_predicted = forecast_mean[-1] - 2 * forecast_std[-1]
        risk_level = 'high' if min_predicted < 0 else 'low'
        
        insights = {
            'current_balance': float(current_balance),
            'predicted_balance_7d': float(predicted_balance),
            'expected_change': float(balance_change),
            'trend': trend,
            'volatility': volatility_level,
            'risk_level': risk_level,
            'confidence_interval': {
                'lower': float(forecast_mean[-1] - 2 * forecast_std[-1]),
                'upper': float(forecast_mean[-1] + 2 * forecast_std[-1])
            }
        }
        
        # Add warnings if needed
        warnings = []
        if min_predicted < 0:
            warnings.append('Potential negative balance within forecast period')
        if volatility_level == 'high':
            warnings.append('High uncertainty in forecast due to variable spending patterns')
        
        if warnings:
            insights['warnings'] = warnings
        
        return insights
    
    def get_spending_patterns(self, account_id: int) -> Dict:
        """
        Analyze spending patterns for better context
        """
        from dashboard.models import Transaction
        from django.db.models import Sum, Avg, Count
        from django.db.models.functions import TruncWeek, TruncMonth
        
        # Get recent transactions
        end_date = timezone.now()
        start_date = end_date - timedelta(days=90)
        
        transactions = Transaction.objects.filter(
            account_id=account_id,
            date__gte=start_date,
            data_source='plaid'
        )
        
        # Weekly patterns
        weekly_spending = transactions.filter(
            amount__lt=0  # Only expenses
        ).annotate(
            week=TruncWeek('date')
        ).values('week').annotate(
            total=Sum('amount'),
            count=Count('id')
        ).order_by('week')
        
        # Category breakdown
        category_spending = transactions.filter(
            amount__lt=0,
            ai_category__isnull=False
        ).values('ai_category').annotate(
            total=Sum('amount'),
            avg=Avg('amount'),
            count=Count('id')
        ).order_by('total')
        
        return {
            'weekly_spending': list(weekly_spending),
            'category_breakdown': list(category_spending),
            'total_transactions': transactions.count(),
            'avg_daily_spending': float(
                transactions.filter(amount__lt=0).aggregate(
                    total=Sum('amount')
                )['total'] or 0
            ) / 90
        }


# Singleton instance
_forecasting_service = None

def get_forecasting_service() -> ChronosForecastingService:
    """Get or create the singleton forecasting service"""
    global _forecasting_service
    if _forecasting_service is None:
        _forecasting_service = ChronosForecastingService()
    return _forecasting_service