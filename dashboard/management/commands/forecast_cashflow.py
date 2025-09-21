"""
Management command to forecast cash flow using Chronos-T5
File: dashboard/management/commands/forecast_cashflow.py
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Account
from dashboard.ml_services.forecasting_service import get_forecasting_service
import json

class Command(BaseCommand):
    help = 'Forecast cash flow using Chronos-T5 model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            default='demo_user',
            help='Username to forecast for'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Number of days to forecast ahead (default: 7)'
        )
        parser.add_argument(
            '--history',
            type=int,
            default=90,
            help='Days of history to use (default: 90)'
        )
        parser.add_argument(
            '--samples',
            type=int,
            default=20,
            help='Number of forecast samples for uncertainty (default: 20)'
        )
        parser.add_argument(
            '--show-patterns',
            action='store_true',
            help='Show spending patterns analysis'
        )
    
    def handle(self, *args, **options):
        username = options['user']
        horizon = options['days']
        history = options['history']
        samples = options['samples']
        show_patterns = options['show_patterns']
        
        self.stdout.write("🚀 Starting Chronos-T5 Cash Flow Forecasting...")
        
        # Get user and account
        try:
            user = User.objects.get(username=username)
            account = user.accounts.first()
            if not account:
                self.stdout.write(self.style.ERROR(f"No account found for user {username}"))
                return
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"User {username} not found"))
            return
        
        self.stdout.write(f"📊 Account: {account.account_name}")
        self.stdout.write(f"   Current Balance: ${account.balance:,.2f}")
        
        # Initialize service
        service = get_forecasting_service()
        
        # Show spending patterns if requested
        if show_patterns:
            self.stdout.write("\n📈 Analyzing spending patterns...")
            patterns = service.get_spending_patterns(account.id)
            
            self.stdout.write(f"\n   Last 90 days statistics:")
            self.stdout.write(f"   • Total transactions: {patterns['total_transactions']}")
            self.stdout.write(f"   • Avg daily spending: ${abs(patterns['avg_daily_spending']):.2f}")
            
            if patterns['category_breakdown']:
                self.stdout.write(f"\n   Top spending categories:")
                for cat in patterns['category_breakdown'][:5]:
                    self.stdout.write(
                        f"   • {cat['ai_category']:20s}: "
                        f"${abs(cat['total']):.2f} ({cat['count']} transactions)"
                    )
        
        # Generate forecast
        self.stdout.write(f"\n🔮 Generating {horizon}-day forecast...")
        self.stdout.write(f"   Using {history} days of historical data")
        self.stdout.write(f"   Generating {samples} forecast samples for uncertainty quantification")
        
        result = service.forecast_balance(
            account_id=account.id,
            horizon=horizon,
            history_days=history,
            num_samples=samples
        )
        
        if 'error' in result:
            self.stdout.write(self.style.ERROR(f"❌ Forecast failed: {result['error']}"))
            return
        
        # Display results
        self.stdout.write(self.style.SUCCESS("\n✅ Forecast generated successfully!"))
        
        # Show forecast summary
        insights = result['insights']
        forecast_data = result['forecast']
        
        self.stdout.write(f"\n📊 Forecast Summary:")
        self.stdout.write(f"   Current Balance:     ${insights['current_balance']:,.2f}")
        self.stdout.write(f"   Predicted (7 days):  ${insights['predicted_balance_7d']:,.2f}")
        self.stdout.write(f"   Expected Change:     ${insights['expected_change']:+,.2f}")
        self.stdout.write(f"   Trend:              {insights['trend'].capitalize()}")
        self.stdout.write(f"   Volatility:         {insights['volatility'].capitalize()}")
        self.stdout.write(f"   Risk Level:         {insights['risk_level'].capitalize()}")
        
        # Show confidence interval
        ci = insights['confidence_interval']
        self.stdout.write(f"\n   95% Confidence Interval:")
        self.stdout.write(f"   • Lower bound: ${ci['lower']:,.2f}")
        self.stdout.write(f"   • Upper bound: ${ci['upper']:,.2f}")
        
        # Show warnings if any
        if 'warnings' in insights:
            self.stdout.write(self.style.WARNING("\n⚠️  Warnings:"))
            for warning in insights['warnings']:
                self.stdout.write(f"   • {warning}")
        
        # Show daily forecast
        self.stdout.write(f"\n📅 Daily Forecast:")
        for i, date in enumerate(forecast_data['dates']):
            mean = forecast_data['mean'][i]
            lower = forecast_data['lower_bound'][i]
            upper = forecast_data['upper_bound'][i]
            
            # Format date nicely
            from datetime import datetime
            dt = datetime.fromisoformat(date)
            date_str = dt.strftime('%a, %b %d')
            
            self.stdout.write(
                f"   {date_str}: ${mean:,.2f} "
                f"(${lower:,.2f} - ${upper:,.2f})"
            )
        
        # Save forecast to file if large
        if horizon > 7:
            filename = f"forecast_{account.id}_{horizon}d.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            self.stdout.write(f"\n💾 Full forecast saved to {filename}")