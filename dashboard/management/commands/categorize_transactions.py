"""
Management command to categorize transactions using SetFit
File: dashboard/management/commands/categorize_transactions.py
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Transaction, Account
from dashboard.ml_services.categorization_service import get_categorization_service
import json

class Command(BaseCommand):
    help = 'Categorize transactions using SetFit model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            default='demo_user',
            help='Username to categorize transactions for'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of transactions to categorize'
        )
        parser.add_argument(
            '--show-stats',
            action='store_true',
            help='Show category statistics after categorization'
        )
        parser.add_argument(
            '--test-single',
            action='store_true',
            help='Test with a single transaction first'
        )
    
    def handle(self, *args, **options):
        username = options['user']
        limit = options['limit']
        show_stats = options['show_stats']
        test_single = options['test_single']
        
        self.stdout.write("🚀 Starting SetFit Transaction Categorization...")
        
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
        
        # Initialize service
        service = get_categorization_service()
        
        # Test single transaction first if requested
        if test_single:
            self.stdout.write("\n📝 Testing single transaction...")
            test_txn = Transaction.objects.filter(
                account=account,
                data_source='plaid'
            ).first()
            
            if test_txn:
                category, confidence = service.predict_single(
                    test_txn.description,
                    test_txn.merchant
                )
                
                self.stdout.write(f"   Description: {test_txn.description}")
                self.stdout.write(f"   Merchant: {test_txn.merchant}")
                self.stdout.write(f"   Amount: ${test_txn.amount}")
                self.stdout.write(f"   → Predicted: {category}")
                self.stdout.write(f"   → Confidence: {confidence:.2%}")
                
                # Update the test transaction
                test_txn.ai_category = category
                test_txn.confidence_score = confidence
                test_txn.save()
                
                self.stdout.write(self.style.SUCCESS("✅ Test successful!"))
            return
        
        # Get uncategorized transactions
        query = Transaction.objects.filter(
            account=account,
            ai_category__isnull=True,
            data_source='plaid'
        )
        
        if limit:
            query = query[:limit]
        
        uncategorized_count = query.count()
        
        if uncategorized_count == 0:
            self.stdout.write("No uncategorized transactions found.")
            
            # Check if already categorized
            categorized = Transaction.objects.filter(
                account=account,
                ai_category__isnull=False,
                data_source='plaid'
            ).count()
            
            if categorized > 0:
                self.stdout.write(f"Already categorized: {categorized} transactions")
        else:
            self.stdout.write(f"Found {uncategorized_count} uncategorized transactions")
            
            # Categorize in batches
            transactions = list(query.values('id', 'description', 'merchant'))
            
            self.stdout.write("🤖 Running SetFit categorization...")
            
            # Process in batches of 32
            batch_size = 32
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i+batch_size]
                results = service.predict_batch(batch)
                
                # Update database
                for result in results:
                    if result.get('ai_category'):
                        Transaction.objects.filter(id=result['id']).update(
                            ai_category=result['ai_category'],
                            confidence_score=result['confidence_score']
                        )
                
                processed = min(i + batch_size, len(transactions))
                self.stdout.write(f"   Processed {processed}/{len(transactions)} transactions...")
            
            self.stdout.write(self.style.SUCCESS(f"✅ Categorized {uncategorized_count} transactions!"))
        
        # Show statistics if requested
        if show_stats or uncategorized_count > 0:
            self.stdout.write("\n📊 Category Statistics:")
            stats = service.get_category_statistics(account.id)
            
            self.stdout.write(f"\nTotal categorized: {stats['total_transactions']}")
            self.stdout.write(f"Average confidence: {stats['average_confidence']:.2%}")
            
            self.stdout.write("\nCategory breakdown:")
            for cat_stat in stats['categories']:
                cat_name = cat_stat['ai_category']
                count = cat_stat['count']
                total = abs(cat_stat['total_amount'] or 0)
                avg = abs(cat_stat['avg_amount'] or 0)
                conf = cat_stat['avg_confidence'] or 0
                
                self.stdout.write(
                    f"  {cat_name:25s}: {count:4d} txns, "
                    f"${total:8.2f} total, ${avg:6.2f} avg, "
                    f"{conf:.1%} confidence"
                )
        
        # Show some examples
        self.stdout.write("\n🔍 Sample categorized transactions:")
        samples = Transaction.objects.filter(
            account=account,
            ai_category__isnull=False,
            data_source='plaid'
        ).order_by('-confidence_score')[:5]
        
        for txn in samples:
            self.stdout.write(
                f"  {txn.description[:40]:40s} → {txn.ai_category:20s} "
                f"({txn.confidence_score:.1%})"
            )