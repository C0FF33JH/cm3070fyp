"""
Management command to test Claude orchestration
File: dashboard/management/commands/test_chat.py
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.models import Account
from dashboard.ml_services.orchestration_service import get_orchestration_service
import json

class Command(BaseCommand):
    help = 'Test Claude orchestration with sample queries'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--query',
            type=str,
            help='Custom query to test'
        )
        parser.add_argument(
            '--test-all',
            action='store_true',
            help='Test with multiple sample queries'
        )
    
    def handle(self, *args, **options):
        custom_query = options.get('query')
        test_all = options.get('test_all')
        
        self.stdout.write("🤖 Testing Claude Orchestration Service...")
        
        # Get user and account
        user = User.objects.get(username='demo_user')
        account = user.accounts.first()
        
        if not account:
            self.stdout.write(self.style.ERROR("No account found for demo_user"))
            return
        
        # Initialize orchestration service
        service = get_orchestration_service()
        
        # Check if API key is set
        if not service.client:
            self.stdout.write(self.style.WARNING(
                "⚠️  No ANTHROPIC_API_KEY set. Using rule-based routing mode.\n"
                "   To enable Claude AI, set your API key:\n"
                "   export ANTHROPIC_API_KEY='your-key-here'"
            ))
        else:
            self.stdout.write(self.style.SUCCESS("✅ Claude AI client initialized"))
        
        # Test queries
        if test_all:
            test_queries = [
                "What's my current balance?",
                "How much did I spend on food last month?",
                "Will I have enough money next week?",
                "Are there any suspicious transactions?",
                "Give me a spending breakdown by category",
                "What's my financial forecast for the next 7 days?",
                "Analyze my spending patterns",
                "Do I have any anomalous transactions?",
            ]
        elif custom_query:
            test_queries = [custom_query]
        else:
            test_queries = [
                "Give me a comprehensive overview of my finances",
                "What are my top spending categories and predict my balance for next week?"
            ]
        
        # Process each query
        for i, query in enumerate(test_queries, 1):
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Query {i}: {query}")
            self.stdout.write(f"{'='*60}")
            
            # Process query
            response = service.process_query(
                user_query=query,
                account_id=account.id
            )
            
            # Display response
            if response.get('success'):
                self.stdout.write(self.style.SUCCESS("✅ Query processed successfully"))
                
                # Natural language response
                if response.get('natural_language_response'):
                    self.stdout.write(f"\n💬 Response:")
                    self.stdout.write(f"   {response['natural_language_response']}")
                
                # Insights
                if response.get('insights'):
                    self.stdout.write(f"\n📊 Insights:")
                    for insight in response['insights']:
                        self.stdout.write(f"   {insight}")
                
                # Suggested actions
                if response.get('suggested_actions'):
                    self.stdout.write(f"\n💡 Suggested Actions:")
                    for action in response['suggested_actions']:
                        self.stdout.write(f"   • {action}")
                
                # Data summary
                if response.get('data'):
                    self.stdout.write(f"\n📈 Data Retrieved:")
                    for key in response['data'].keys():
                        self.stdout.write(f"   • {key}")
                
                # Charts available
                if response.get('charts'):
                    self.stdout.write(f"\n📊 Charts Generated:")
                    for chart in response['charts']:
                        self.stdout.write(f"   • {chart['type']}: {chart['title']}")
            else:
                self.stdout.write(self.style.ERROR("❌ Query processing failed"))
                if response.get('error'):
                    self.stdout.write(f"   Error: {response['error']}")
        
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(self.style.SUCCESS("✅ Orchestration test complete!"))