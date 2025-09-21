"""
Management command to populate Track 1: Open Banking Demo Data
Compatible with SetFit categorization and Chronos-T5 forecasting
"""

import random
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Account, Transaction
from django.utils import timezone

class Command(BaseCommand):
    help = 'Generate 365 days of realistic banking transactions for demo user'
    
    def __init__(self):
        super().__init__()
        random.seed(42)  # Reproducible data
        np.random.seed(42)
        
        # Define realistic merchants by category (matching SetFit training)
        self.merchants = {
            'FOOD_AND_DRINK': [
                ('Starbucks', (3, 12)),
                ('McDonalds', (5, 20)),
                ('Whole Foods Market', (20, 200)),
                ('Kroger', (15, 150)),
                ('Subway', (6, 15)),
                ('Chipotle', (8, 25)),
                ('Local Restaurant', (15, 80)),
                ('7-Eleven', (3, 20)),
            ],
            'TRANSPORTATION': [
                ('Shell', (25, 75)),
                ('Uber', (8, 45)),
                ('Chevron', (30, 80)),
                ('Lyft', (10, 35)),
                ('Public Transit', (2, 10)),
                ('Parking', (2, 20)),
            ],
            'GENERAL_MERCHANDISE': [
                ('Amazon', (10, 300)),
                ('Target', (20, 200)),
                ('Walmart', (15, 250)),
                ('Best Buy', (50, 500)),
                ('Home Depot', (25, 300)),
            ],
            'ENTERTAINMENT': [
                ('Netflix', (15, 16)),  # Subscription
                ('Spotify', (10, 11)),  # Subscription
                ('AMC Theaters', (12, 40)),
                ('Steam Games', (10, 60)),
                ('Ticketmaster', (50, 200)),
            ],
            'RENT_AND_UTILITIES': [
                ('Property Management LLC', (1500, 1500)),  # Rent
                ('Electric Company', (80, 150)),
                ('Internet Provider', (60, 60)),
                ('Water Utility', (30, 50)),
            ],
            'MEDICAL': [
                ('CVS Pharmacy', (10, 100)),
                ('Walgreens', (15, 80)),
                ('City Medical Center', (50, 300)),
                ('Dental Associates', (100, 200)),
            ],
            'PERSONAL_CARE': [
                ('Planet Fitness', (25, 25)),  # Gym
                ('Great Clips', (20, 30)),
                ('Salon & Spa', (40, 120)),
            ],
            'INCOME': [
                ('Employer Direct Deposit', (5000, 5000)),  # Salary
            ],
            'TRANSFER_OUT': [
                ('Zelle Transfer', (20, 200)),
                ('Venmo', (10, 100)),
            ],
        }
        
        # Subscriptions with fixed days
        self.subscriptions = [
            ('Netflix', 'ENTERTAINMENT', 15.99, 5),
            ('Spotify', 'ENTERTAINMENT', 10.99, 10),
            ('Planet Fitness', 'PERSONAL_CARE', 25.00, 1),
            ('Internet Provider', 'RENT_AND_UTILITIES', 59.99, 15),
        ]
    
    def create_description(self, merchant, category, amount):
        """Generate realistic transaction descriptions matching SetFit training format"""
        
        # Templates based on your training data patterns
        templates = {
            'pos': [
                f"POS PURCHASE {merchant.upper()}",
                f"POS {merchant} STORE #{random.randint(1000, 9999)}",
                f"PURCHASE {merchant}",
                f"DEBIT CARD PURCHASE {merchant}",
            ],
            'online': [
                f"ONLINE PURCHASE {merchant}",
                f"WEB PMT {merchant}",
                f"{merchant}.COM",
                f"PAYPAL *{merchant}",
            ],
            'ach': [
                f"ACH DEBIT {merchant}",
                f"RECURRING PMT {merchant}",
                f"AUTOPAY {merchant}",
                f"ELECTRONIC PMT {merchant}",
            ],
            'atm': [
                f"ATM WITHDRAWAL",
                f"CASH WITHDRAWAL ATM #{random.randint(10000, 99999)}",
            ],
            'transfer': [
                f"TRANSFER TO {merchant}",
                f"ONLINE TRANSFER",
                f"MOBILE TRANSFER {merchant}",
            ],
            'ambiguous': [
                f"PURCHASE #{random.randint(1000000, 9999999)}",
                f"CARD {random.randint(1000, 9999)}",
                f"PMT {random.randint(100000, 999999)}",
                f"TRANSACTION {random.randint(100000, 999999)}",
            ]
        }
        
        # 25% chance of ambiguous description (matching your SetFit training)
        if random.random() < 0.25:
            return random.choice(templates['ambiguous'])
        
        # Select template based on category
        if category == 'INCOME':
            return f"DIRECT DEPOSIT {merchant}"
        elif category in ['RENT_AND_UTILITIES', 'ENTERTAINMENT'] and 'subscription' in merchant.lower():
            return random.choice(templates['ach'])
        elif 'Amazon' in merchant or 'Online' in merchant:
            return random.choice(templates['online'])
        elif category in ['TRANSFER_IN', 'TRANSFER_OUT']:
            return random.choice(templates['transfer'])
        else:
            return random.choice(templates['pos'])
    
    def handle(self, *args, **options):
        self.stdout.write("🚀 Generating Open Banking Demo Data (365 days)...")
        
        # Create or get demo user
        user, created = User.objects.get_or_create(
            username='demo_user',
            defaults={
                'email': 'demo@example.com',
                'first_name': 'Demo',
                'last_name': 'User'
            }
        )
        if created:
            user.set_password('demo123')
            user.save()
            self.stdout.write(self.style.SUCCESS(f"✅ Created demo user"))
        
        # Clear existing demo data
        Transaction.objects.filter(account__user=user, data_source='plaid').delete()
        Account.objects.filter(user=user).delete()
        
        # Create demo account
        account = Account.objects.create(
            user=user,
            account_name="Personal Checking",
            balance=Decimal('8000.00'),  # Starting balance
            currency='USD'
        )
        self.stdout.write(self.style.SUCCESS(f"✅ Created account: {account.account_name}"))
        
        # Generate 365 days of transactions
        start_date = timezone.now() - timedelta(days=365)
        transactions = []
        current_balance = 8000.00
        
        # Track daily spending for patterns
        daily_transactions = {}
        
        for day_offset in range(365):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            day_of_month = current_date.day
            is_weekend = day_of_week in [5, 6]
            
            day_transactions = []
            
            # 1. INCOME - First of month
            if day_of_month == 1:
                merchant = 'Employer Direct Deposit'
                amount = 5000.00
                current_balance += amount
                
                day_transactions.append({
                    'date': current_date,
                    'amount': Decimal(str(amount)),
                    'merchant': merchant,
                    'category': 'INCOME',
                    'description': self.create_description(merchant, 'INCOME', amount),
                    'balance': current_balance
                })
            
            # 2. RENT - First of month
            if day_of_month == 1:
                merchant = 'Property Management LLC'
                amount = -1500.00
                current_balance += amount
                
                day_transactions.append({
                    'date': current_date,
                    'amount': Decimal(str(amount)),
                    'merchant': merchant,
                    'category': 'RENT_AND_UTILITIES',
                    'description': 'ACH DEBIT PROPERTY MGMT RENT',
                    'balance': current_balance
                })
            
            # 3. SUBSCRIPTIONS
            for sub_merchant, sub_category, sub_amount, sub_day in self.subscriptions:
                if day_of_month == sub_day:
                    amount = -sub_amount
                    current_balance += amount
                    
                    day_transactions.append({
                        'date': current_date,
                        'amount': Decimal(str(amount)),
                        'merchant': sub_merchant,
                        'category': sub_category,
                        'description': f"RECURRING PMT {sub_merchant.upper()}",
                        'balance': current_balance
                    })
            
            # 4. UTILITIES - Mid month
            if day_of_month == 15:
                # Electric bill
                merchant = 'Electric Company'
                amount = -random.uniform(80, 150)
                current_balance += amount
                
                day_transactions.append({
                    'date': current_date,
                    'amount': Decimal(str(round(amount, 2))),
                    'merchant': merchant,
                    'category': 'RENT_AND_UTILITIES',
                    'description': 'ACH DEBIT ELECTRIC UTILITY',
                    'balance': current_balance
                })
            
            # 5. DAILY SPENDING
            # More transactions on weekends
            if is_weekend:
                num_transactions = random.randint(2, 6)
            else:
                num_transactions = random.randint(0, 4)
            
            # Category weights for realistic spending
            category_weights = {
                'FOOD_AND_DRINK': 0.35,
                'TRANSPORTATION': 0.15,
                'GENERAL_MERCHANDISE': 0.20,
                'ENTERTAINMENT': 0.10,
                'MEDICAL': 0.05,
                'PERSONAL_CARE': 0.05,
                'TRANSFER_OUT': 0.10,
            }
            
            for _ in range(num_transactions):
                # Select category
                category = np.random.choice(
                    list(category_weights.keys()),
                    p=list(category_weights.values())
                )
                
                # Select merchant and amount from category
                if category in self.merchants:
                    merchant_data = random.choice(self.merchants[category])
                    merchant = merchant_data[0]
                    amount_range = merchant_data[1]
                    amount = -random.uniform(*amount_range)
                    
                    # Add some variance
                    if is_weekend:
                        amount *= random.uniform(1.1, 1.3)  # Spend more on weekends
                    
                    current_balance += amount
                    
                    day_transactions.append({
                        'date': current_date,
                        'amount': Decimal(str(round(amount, 2))),
                        'merchant': merchant,
                        'category': category,
                        'description': self.create_description(merchant, category, amount),
                        'balance': current_balance
                    })
            
            # 6. OCCASIONAL ANOMALIES (5-7 total)
            # Create predictable anomalies for demo
            if day_offset in [45, 120, 200, 280, 340]:
                # Large unusual purchase
                merchant = random.choice(['Unknown Merchant', 'Online Store #9999', 'INTL PURCHASE'])
                amount = -random.uniform(500, 2000)
                current_balance += amount
                
                day_transactions.append({
                    'date': current_date,
                    'amount': Decimal(str(round(amount, 2))),
                    'merchant': merchant,
                    'category': 'GENERAL_MERCHANDISE',
                    'description': f"SUSPICIOUS TXN {random.randint(100000, 999999)}",
                    'balance': current_balance,
                    'is_anomaly': True
                })
            
            # Store daily transactions
            transactions.extend(day_transactions)
        
        # Bulk create transactions
        self.stdout.write("💾 Saving transactions to database...")
        
        for i, trans in enumerate(transactions):
            Transaction.objects.create(
                account=account,
                transaction_id=f'DEMO_{i+1:06d}',
                date=trans['date'],
                description=trans['description'],
                merchant=trans['merchant'],
                amount=trans['amount'],
                category=trans.get('category'),
                ai_category=trans.get('category'),  # Pre-populate for demo
                is_anomaly=trans.get('is_anomaly', False),
                confidence_score=random.uniform(0.85, 0.99) if not trans.get('is_anomaly') else 0.0,
                data_source='plaid'
            )
        
        # Update final account balance
        account.balance = Decimal(str(round(current_balance, 2)))
        account.save()
        
        self.stdout.write(self.style.SUCCESS(
            f"\n✅ Generated {len(transactions)} transactions over 365 days"
        ))
        self.stdout.write(f"   Starting balance: $8,000.00")
        self.stdout.write(f"   Final balance: ${account.balance:,.2f}")
        self.stdout.write(f"   Categories: {len(set(t.get('category') for t in transactions if t.get('category')))}")
        self.stdout.write(f"   Unique merchants: {len(set(t['merchant'] for t in transactions))}")
        self.stdout.write(f"   Anomalies created: 5")
        
        # Category distribution
        category_counts = {}
        for trans in transactions:
            cat = trans.get('category', 'UNKNOWN')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        self.stdout.write("\n📊 Category Distribution:")
        for cat, count in sorted(category_counts.items()):
            pct = (count / len(transactions)) * 100
            self.stdout.write(f"   {cat:25s}: {count:4d} ({pct:5.1f}%)")