from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class Account(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='accounts')
    account_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    account_name = models.CharField(max_length=100, default="Main Account")
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    currency = models.CharField(max_length=3, default='USD')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.account_name}"
    
    class Meta:
        ordering = ['-created_at']

class Transaction(models.Model):
    # Plaid-compatible categories (16 primary categories)
    CATEGORY_CHOICES = [
        ('FOOD_AND_DRINK', 'Food & Drink'),
        ('GENERAL_MERCHANDISE', 'General Merchandise'),
        ('TRANSPORTATION', 'Transportation'),
        ('ENTERTAINMENT', 'Entertainment'),
        ('GENERAL_SERVICES', 'General Services'),
        ('MEDICAL', 'Medical'),
        ('PERSONAL_CARE', 'Personal Care'),
        ('HOME_IMPROVEMENT', 'Home Improvement'),
        ('RENT_AND_UTILITIES', 'Rent & Utilities'),
        ('TRAVEL', 'Travel'),
        ('LOAN_PAYMENTS', 'Loan Payments'),
        ('GOVERNMENT_AND_NON_PROFIT', 'Government & Non-Profit'),
        ('INCOME', 'Income'),
        ('TRANSFER_IN', 'Transfer In'),
        ('TRANSFER_OUT', 'Transfer Out'),
        ('BANK_FEES', 'Bank Fees'),
    ]
    
    # Core transaction fields (Plaid-compatible)
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='transactions')
    transaction_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    date = models.DateTimeField(default=timezone.now)
    description = models.TextField()  # Maps to Plaid 'name' field
    merchant = models.CharField(max_length=200, blank=True, null=True)  # merchant_name in Plaid
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Category fields
    category = models.CharField(
        max_length=50, 
        choices=CATEGORY_CHOICES, 
        blank=True, 
        null=True
    )
    ai_category = models.CharField(
        max_length=50,
        choices=CATEGORY_CHOICES,
        blank=True, 
        null=True
    )  # AI-predicted category
    confidence_score = models.FloatField(default=0.0)  # Model confidence
    
    # Transaction type
    transaction_type = models.CharField(
        max_length=10,
        choices=[('debit', 'Debit'), ('credit', 'Credit')],
        blank=True,
        null=True
    )
    
    # Anomaly detection
    is_anomaly = models.BooleanField(default=False)
    anomaly_score = models.FloatField(default=0.0)
    
    # Additional Kaggle fields for fraud detection (Track 2)
    # These will be null for Plaid-like transactions
    cc_num = models.CharField(max_length=20, blank=True, null=True)
    first = models.CharField(max_length=50, blank=True, null=True)
    last = models.CharField(max_length=50, blank=True, null=True)
    gender = models.CharField(max_length=1, blank=True, null=True)
    street = models.CharField(max_length=200, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=2, blank=True, null=True)
    zip = models.CharField(max_length=10, blank=True, null=True)
    lat = models.FloatField(blank=True, null=True)
    long = models.FloatField(blank=True, null=True)
    city_pop = models.IntegerField(blank=True, null=True)
    job = models.CharField(max_length=100, blank=True, null=True)
    dob = models.DateField(blank=True, null=True)
    trans_num = models.CharField(max_length=100, blank=True, null=True)
    unix_time = models.BigIntegerField(blank=True, null=True)
    merch_lat = models.FloatField(blank=True, null=True)
    merch_long = models.FloatField(blank=True, null=True)
    
    # Data source flag to distinguish between tracks
    data_source = models.CharField(
        max_length=20,
        choices=[
            ('plaid', 'Open Banking'),
            ('kaggle', 'Enhanced Data')
        ],
        default='plaid'
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.date.strftime('%Y-%m-%d')} - {self.description[:30]} - ${self.amount}"
    
    def save(self, *args, **kwargs):
        # Auto-determine transaction type based on amount if not set
        if self.transaction_type is None:
            if self.amount > 0:
                self.transaction_type = 'credit'
            else:
                self.transaction_type = 'debit'
        
        # Auto-set category for income transactions
        if self.amount > 0 and self.category is None:
            if 'deposit' in self.description.lower() or 'payroll' in self.description.lower():
                self.category = 'INCOME'
            elif 'transfer' in self.description.lower() or 'zelle' in self.description.lower():
                self.category = 'TRANSFER_IN'
        
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-date']
        indexes = [
            models.Index(fields=['-date']),
            models.Index(fields=['account', '-date']),
            models.Index(fields=['category']),
            models.Index(fields=['is_anomaly']),
            models.Index(fields=['data_source']),
        ]

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    message = models.TextField()
    response = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']

class AnalysisResult(models.Model):
    ANALYSIS_TYPES = [
        ('categorization', 'Transaction Categorization'),
        ('fraud', 'Fraud Detection'),
        ('cashflow', 'Cash Flow Forecast'),
        ('comprehensive', 'Comprehensive Analysis'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    analysis_type = models.CharField(max_length=50, choices=ANALYSIS_TYPES)
    result_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_analysis_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']

class DailyBalance(models.Model):
    """For cash flow forecasting - stores daily aggregated balances"""
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='daily_balances')
    date = models.DateField()
    opening_balance = models.DecimalField(max_digits=12, decimal_places=2)
    closing_balance = models.DecimalField(max_digits=12, decimal_places=2)
    total_credits = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_debits = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    transaction_count = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.account.account_name} - {self.date} - ${self.closing_balance}"
    
    class Meta:
        ordering = ['-date']
        unique_together = ['account', 'date']
        indexes = [
            models.Index(fields=['account', '-date']),
        ]

class ModelPrediction(models.Model):
    """Store model predictions for audit and improvement"""
    MODEL_CHOICES = [
        ('setfit', 'SetFit Categorization'),
        ('xgboost', 'XGBoost Fraud Detection'),
        ('chronos', 'Chronos-T5 Forecasting'),
    ]
    
    transaction = models.ForeignKey(
        Transaction, 
        on_delete=models.CASCADE, 
        related_name='predictions',
        null=True,
        blank=True
    )
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    prediction = models.JSONField()
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_model_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']