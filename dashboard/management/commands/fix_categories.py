"""
Management command to fix miscategorized transactions
File: dashboard/management/commands/fix_categories.py
"""

from django.core.management.base import BaseCommand
from dashboard.models import Transaction

class Command(BaseCommand):
    help = 'Fix transactions with integer categories'
    
    def handle(self, *args, **options):
        # Mapping from integers to categories
        label_map = {
            '0': 'FOOD_AND_DRINK',
            '1': 'GENERAL_MERCHANDISE',
            '2': 'TRANSPORTATION',
            '3': 'ENTERTAINMENT',
            '4': 'GENERAL_SERVICES',
            '5': 'MEDICAL',
            '6': 'PERSONAL_CARE',
            '7': 'HOME_IMPROVEMENT',
            '8': 'RENT_AND_UTILITIES',
            '9': 'TRAVEL',
            '10': 'LOAN_PAYMENTS',
            '11': 'GOVERNMENT_AND_NON_PROFIT',
            '12': 'INCOME',
            '13': 'TRANSFER_IN',
            '14': 'TRANSFER_OUT',
            '15': 'BANK_FEES'
        }
        
        self.stdout.write("🔧 Fixing integer categories...")
        
        fixed_count = 0
        for int_label, category in label_map.items():
            count = Transaction.objects.filter(ai_category=int_label).update(ai_category=category)
            if count > 0:
                self.stdout.write(f"   Fixed {count} transactions: {int_label} → {category}")
                fixed_count += count
        
        if fixed_count > 0:
            self.stdout.write(self.style.SUCCESS(f"✅ Fixed {fixed_count} miscategorized transactions"))
        else:
            self.stdout.write("No transactions with integer categories found")
        
        # Show current statistics
        from dashboard.ml_services import get_categorization_service
        from django.contrib.auth.models import User
        
        user = User.objects.get(username='demo_user')
        account = user.accounts.first()
        
        service = get_categorization_service()
        stats = service.get_category_statistics(account.id)
        
        self.stdout.write("\n📊 Updated Category Statistics:")
        self.stdout.write(f"Total categorized: {stats['total_transactions']}")
        self.stdout.write(f"Average confidence: {stats['average_confidence']:.2%}")
        
        self.stdout.write("\nCategory breakdown:")
        for cat_stat in stats['categories']:
            cat_name = cat_stat['ai_category']
            count = cat_stat['count']
            total = abs(cat_stat['total_amount'] or 0)
            
            self.stdout.write(
                f"  {cat_name:25s}: {count:4d} txns, ${total:8.2f} total"
            )