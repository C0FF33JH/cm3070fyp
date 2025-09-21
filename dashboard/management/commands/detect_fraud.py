"""
Management command to detect fraud using XGBoost
File: dashboard/management/commands/detect_fraud.py
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Transaction, Account
from dashboard.ml_services.fraud_service import get_fraud_service
import json

class Command(BaseCommand):
    help = 'Detect fraudulent transactions using XGBoost model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            default='fraud_analyst',
            help='Username to analyze (default: fraud_analyst for Kaggle data)'
        )
        parser.add_argument(
            '--test-single',
            action='store_true',
            help='Test with a single transaction first'
        )
        parser.add_argument(
            '--analyze-plaid',
            action='store_true',
            help='Run anomaly detection on Plaid data using statistical methods'
        )
        parser.add_argument(
            '--sensitivity',
            type=float,
            default=2.0,
            help='Sensitivity for statistical anomaly detection (default: 2.0 std devs)'
        )
    
    def handle(self, *args, **options):
        username = options['user']
        test_single = options['test_single']
        analyze_plaid = options['analyze_plaid']
        sensitivity = options['sensitivity']
        
        self.stdout.write("🚀 Starting XGBoost Fraud Detection...")
        
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
        service = get_fraud_service()
        
        # Handle Plaid anomaly detection
        if analyze_plaid:
            self.stdout.write("\n🔍 Running statistical anomaly detection on Plaid data...")
            
            # Switch to demo_user for Plaid data
            demo_user = User.objects.get(username='demo_user')
            demo_account = demo_user.accounts.first()
            
            results = service.detect_anomalies_plaid(demo_account.id, sensitivity)
            
            self.stdout.write(f"\n📊 Anomaly Detection Results:")
            self.stdout.write(f"   Total analyzed: {results['total_analyzed']}")
            self.stdout.write(f"   Anomalies found: {results['anomalies_found']}")
            self.stdout.write(f"   Method: Z-score (>{sensitivity} std devs)")
            
            if results['anomalies']:
                self.stdout.write("\n🚨 Top Anomalous Transactions:")
                for anom in results['anomalies'][:5]:
                    self.stdout.write(
                        f"   • {anom['date'][:10]}: {anom['description'][:30]:30s} "
                        f"${abs(anom['amount']):8.2f} "
                        f"(Z-score: {anom['z_score']:.1f})"
                    )
            return
        
        # Test single Kaggle transaction
        if test_single:
            self.stdout.write("\n📝 Testing single Kaggle transaction...")
            
            # Get a fraud transaction for testing
            test_trans = Transaction.objects.filter(
                account__user=user,
                data_source='kaggle',
                is_anomaly=True  # Get an actual fraud case
            ).first()
            
            if not test_trans:
                # Fall back to any Kaggle transaction
                test_trans = Transaction.objects.filter(
                    account__user=user,
                    data_source='kaggle'
                ).first()
            
            if test_trans:
                trans_data = {
                    'amount': test_trans.amount,
                    'lat': test_trans.lat,
                    'long': test_trans.long,
                    'city_pop': test_trans.city_pop,
                    'merch_lat': test_trans.merch_lat,
                    'merch_long': test_trans.merch_long,
                    'dob': test_trans.dob,
                    'date': test_trans.date,
                    'category': test_trans.category,
                    'gender': test_trans.gender,
                    'state': test_trans.state
                }
                
                is_fraud, prob, details = service.predict_fraud(trans_data)
                
                self.stdout.write(f"\n   Transaction Details:")
                self.stdout.write(f"   • Description: {test_trans.description}")
                self.stdout.write(f"   • Merchant: {test_trans.merchant}")
                self.stdout.write(f"   • Amount: ${abs(test_trans.amount):.2f}")
                self.stdout.write(f"   • Actual fraud: {test_trans.is_anomaly}")
                
                self.stdout.write(f"\n   Prediction Results:")
                self.stdout.write(f"   • Predicted fraud: {is_fraud}")
                self.stdout.write(f"   • Probability: {prob:.2%}")
                
                if 'risk_level' in details:
                    self.stdout.write(f"   • Risk level: {details['risk_level']}")
                
                if details.get('risk_factors'):
                    self.stdout.write(f"   • Top risk factors: {', '.join(details['risk_factors'][:3])}")
                
                if 'error' in details:
                    self.stdout.write(self.style.WARNING(f"   ⚠️ Error: {details['error']}"))
                
                if is_fraud == test_trans.is_anomaly:
                    self.stdout.write(self.style.SUCCESS("   ✅ Correct prediction!"))
                else:
                    self.stdout.write(self.style.WARNING("   ⚠️ Incorrect prediction"))
            return
        
        # Full batch analysis of Kaggle data
        self.stdout.write(f"\n🔍 Analyzing fraud in Kaggle transactions...")
        self.stdout.write(f"   User: {username}")
        
        # Check data availability
        kaggle_count = Transaction.objects.filter(
            account__user=user,
            data_source='kaggle'
        ).count()
        
        if kaggle_count == 0:
            self.stdout.write(self.style.ERROR(
                f"\n❌ No Kaggle transactions found for {username}"
            ))
            self.stdout.write(
                "   Tip: Run 'python manage.py populate_fraud' first to load Kaggle data"
            )
            return
        
        self.stdout.write(f"   Found {kaggle_count} Kaggle transactions to analyze")
        
        # Run batch analysis
        results = service.analyze_batch()
        
        if 'error' in results:
            self.stdout.write(self.style.ERROR(f"❌ Analysis failed: {results['error']}"))
            return
        
        # Display results
        self.stdout.write(self.style.SUCCESS("\n✅ Fraud analysis completed!"))
        
        # Summary statistics
        self.stdout.write(f"\n📊 Analysis Summary:")
        self.stdout.write(f"   Total transactions: {results['total_transactions']}")
        self.stdout.write(f"   Actual fraud cases: {results['actual_fraud_count']}")
        self.stdout.write(f"   Predicted fraud cases: {results['predicted_fraud_count']}")
        
        # Model performance metrics
        metrics = results['metrics']
        self.stdout.write(f"\n📈 Model Performance:")
        self.stdout.write(f"   • Accuracy:  {metrics['accuracy']:.2%}")
        self.stdout.write(f"   • Precision: {metrics['precision']:.2%}")
        self.stdout.write(f"   • Recall:    {metrics['recall']:.2%}")
        self.stdout.write(f"   • F1 Score:  {metrics['f1_score']:.2%}")
        
        # Confusion matrix
        cm = results['confusion_matrix']
        self.stdout.write(f"\n🎯 Confusion Matrix:")
        self.stdout.write(f"   True Positives:  {cm['true_positives']:3d} (Correctly identified fraud)")
        self.stdout.write(f"   True Negatives:  {cm['true_negatives']:3d} (Correctly identified normal)")
        self.stdout.write(f"   False Positives: {cm['false_positives']:3d} (Normal flagged as fraud)")
        self.stdout.write(f"   False Negatives: {cm['false_negatives']:3d} (Fraud missed)")
        
        # Risk distribution
        dist = results['risk_distribution']
        self.stdout.write(f"\n📊 Risk Distribution:")
        for level in ['very_low', 'low', 'medium', 'high', 'very_high']:
            count = dist[level]
            bar = '█' * (count // 2) if count > 0 else ''
            self.stdout.write(f"   {level:10s}: {count:3d} {bar}")
        
        # Top risks
        if results['top_risks']:
            self.stdout.write(f"\n🚨 Top Risk Transactions:")
            for i, risk in enumerate(results['top_risks'][:5], 1):
                status = "✓" if risk['predicted_fraud'] == risk['actual_fraud'] else "✗"
                actual = "FRAUD" if risk['actual_fraud'] else "NORMAL"
                self.stdout.write(
                    f"   {i}. {risk['merchant'][:25]:25s} "
                    f"${abs(risk['amount']):8.2f} "
                    f"Risk: {risk['probability']:.1%} "
                    f"[{actual}] {status}"
                )