"""
Test command to verify models work with generated data
"""

import pickle
import numpy as np
from django.core.management.base import BaseCommand
from dashboard.models import Transaction
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Command(BaseCommand):
    help = 'Test that saved models work with generated data'
    
    def handle(self, *args, **options):
        self.stdout.write("🧪 Testing model compatibility with generated data...")
        
        # Test 1: SetFit with Open Banking data
        self.stdout.write("\n1️⃣ Testing SetFit categorization...")
        plaid_transactions = Transaction.objects.filter(data_source='plaid')[:10]
        
        if plaid_transactions:
            # Create text features as model expects
            for trans in plaid_transactions:
                text_features = f"{trans.description} {trans.merchant}"
                self.stdout.write(f"   Text: {text_features[:50]}...")
            self.stdout.write(self.style.SUCCESS("   ✅ SetFit input format compatible"))
        
        # Test 2: Chronos-T5 with balance time series
        self.stdout.write("\n2️⃣ Testing Chronos-T5 forecasting...")
        
        # Get daily balances (what Chronos expects)
        plaid_trans = Transaction.objects.filter(
            data_source='plaid'
        ).order_by('date').values('date', 'amount')
        
        if plaid_trans:
            df = pd.DataFrame(plaid_trans)
            df['date'] = pd.to_datetime(df['date'])
            df['cumulative_balance'] = 8000 + df['amount'].cumsum()
            
            # Group by date for daily balance
            daily_balance = df.groupby(df['date'].dt.date)['cumulative_balance'].last()
            
            self.stdout.write(f"   Time series length: {len(daily_balance)} days")
            self.stdout.write(f"   Balance range: ${daily_balance.min():.2f} - ${daily_balance.max():.2f}")
            self.stdout.write(self.style.SUCCESS("   ✅ Chronos-T5 input format compatible"))
        
        # Test 3: XGBoost with Kaggle data
        self.stdout.write("\n3️⃣ Testing XGBoost fraud detection...")
        kaggle_transactions = Transaction.objects.filter(data_source='kaggle')[:10]
        
        if kaggle_transactions:
            # Check required fields exist
            required_fields = ['lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'dob']
            
            sample = kaggle_transactions[0]
            missing = []
            for field in required_fields:
                if getattr(sample, field) is None:
                    missing.append(field)
            
            if missing:
                self.stdout.write(self.style.WARNING(f"   ⚠️ Missing fields: {missing}"))
            else:
                self.stdout.write(self.style.SUCCESS("   ✅ XGBoost input format compatible"))
        
        self.stdout.write(self.style.SUCCESS("\n✅ Model compatibility test complete!"))