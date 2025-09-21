"""
Management command to populate Track 2: Kaggle Fraud Data
Compatible with XGBoost fraud detection model
"""

import pandas as pd
import random
from datetime import datetime
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Account, Transaction
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Load sample of Kaggle fraud dataset for XGBoost demo'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-path',
            type=str,
            default='fraudTest.csv',
            help='Path to Kaggle fraud test CSV file'
        )
        parser.add_argument(
            '--sample-size',
            type=int,
            default=5000,
            help='Number of transactions to load'
        )
    
    def handle(self, *args, **options):
        csv_path = options['csv_path']
        sample_size = options['sample_size']
        
        self.stdout.write(f"🚀 Loading Kaggle Fraud Data from {csv_path}...")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(
                f"❌ File not found: {csv_path}\n"
                f"   Please download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection"
            ))
            return
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path, index_col=0)
            self.stdout.write(f"✅ Loaded {len(df):,} transactions from CSV")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error loading CSV: {e}"))
            return
        
        # Sample data
        if len(df) > sample_size:
            # Stratified sample to maintain fraud ratio
            fraud_df = df[df['is_fraud'] == 1]
            normal_df = df[df['is_fraud'] == 0]
            
            # Keep fraud ratio
            n_fraud = min(len(fraud_df), int(sample_size * 0.01))  # ~1% fraud
            n_normal = sample_size - n_fraud
            
            sampled_df = pd.concat([
                fraud_df.sample(n=n_fraud, random_state=42),
                normal_df.sample(n=n_normal, random_state=42)
            ]).reset_index(drop=True)
            
            self.stdout.write(f"📊 Sampled {len(sampled_df)} transactions "
                            f"({n_fraud} fraud, {n_normal} normal)")
        else:
            sampled_df = df
        
        # Create fraud demo user
        user, created = User.objects.get_or_create(
            username='fraud_analyst',
            defaults={
                'email': 'analyst@example.com',
                'first_name': 'Fraud',
                'last_name': 'Analyst'
            }
        )
        if created:
            user.set_password('fraud123')
            user.save()
            self.stdout.write(self.style.SUCCESS(f"✅ Created fraud analyst user"))
        
        # Clear existing fraud data
        Transaction.objects.filter(data_source='kaggle').delete()
        
        # Create accounts for unique card numbers
        self.stdout.write("Creating accounts for card holders...")
        cc_nums = sampled_df['cc_num'].unique()[:10]  # Limit to 10 accounts for demo
        
        accounts = {}
        for cc_num in cc_nums:
            card_data = sampled_df[sampled_df['cc_num'] == cc_num].iloc[0]
            
            # Create account for this card
            account, _ = Account.objects.get_or_create(
                user=user,
                account_id=f"CC_{str(cc_num)[-4:]}",
                defaults={
                    'account_name': f"{card_data['first']} {card_data['last']} - ****{str(cc_num)[-4:]}",
                    'balance': 10000,  # Default balance
                    'currency': 'USD'
                }
            )
            accounts[cc_num] = account
        
        # Load transactions
        self.stdout.write("💾 Loading transactions into database...")
        
        transactions_created = 0
        fraud_count = 0
        
        for idx, row in sampled_df.iterrows():
            # Only load transactions for accounts we created
            if row['cc_num'] not in accounts:
                continue
            
            account = accounts[row['cc_num']]
            
            # Parse date
            trans_date = pd.to_datetime(row['trans_date_trans_time'])
            
            # Clean merchant name (remove 'fraud_' prefix if present)
            merchant = row['merchant'].replace('fraud_', '')
            
            # Create transaction with all Kaggle fields
            Transaction.objects.create(
                account=account,
                transaction_id=f"KAGGLE_{row['trans_num']}",
                date=trans_date,
                description=f"CARD PURCHASE {merchant.upper()}",
                merchant=merchant,
                amount=-abs(row['amt']),  # Make negative for spending
                category=row['category'].upper().replace('_', '_AND_') 
                        if '_' in row['category'] else row['category'].upper(),
                is_anomaly=bool(row['is_fraud']),
                
                # Kaggle-specific fields for XGBoost
                cc_num=str(row['cc_num']),
                first=row['first'],
                last=row['last'],
                gender=row['gender'],
                street=row['street'],
                city=row['city'],
                state=row['state'],
                zip=str(row['zip']),
                lat=row['lat'],
                long=row['long'],
                city_pop=int(row['city_pop']),
                job=row['job'],
                dob=pd.to_datetime(row['dob']),
                trans_num=row['trans_num'],
                unix_time=int(row['unix_time']),
                merch_lat=row['merch_lat'],
                merch_long=row['merch_long'],
                
                data_source='kaggle'
            )
            
            transactions_created += 1
            if row['is_fraud']:
                fraud_count += 1
            
            if transactions_created % 500 == 0:
                self.stdout.write(f"   Loaded {transactions_created} transactions...")
        
        self.stdout.write(self.style.SUCCESS(
            f"\n✅ Successfully loaded Kaggle fraud dataset!"
        ))
        self.stdout.write(f"   Total transactions: {transactions_created}")
        self.stdout.write(f"   Fraud transactions: {fraud_count}")
        self.stdout.write(f"   Normal transactions: {transactions_created - fraud_count}")
        self.stdout.write(f"   Fraud rate: {(fraud_count/transactions_created)*100:.2f}%")
        self.stdout.write(f"   Accounts created: {len(accounts)}")