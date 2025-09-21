"""
Management command to test receipt parsing with TrOCR
File: dashboard/management/commands/parse_receipt.py
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from dashboard.ml_services.receipt_service import get_receipt_service
from PIL import Image, ImageDraw, ImageFont
import json
import os

class Command(BaseCommand):
    help = 'Parse receipt images using TrOCR model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--image',
            type=str,
            help='Path to receipt image file'
        )
        parser.add_argument(
            '--generate-sample',
            action='store_true',
            help='Generate a sample receipt image for testing'
        )
        parser.add_argument(
            '--generate-all',
            action='store_true',
            help='Generate all types of sample receipts'
        )
        parser.add_argument(
            '--receipt-type',
            type=str,
            default='grocery',
            choices=['grocery', 'coffee', 'restaurant', 'pharmacy'],
            help='Type of receipt to generate (default: grocery)'
        )
        parser.add_argument(
            '--match-transaction',
            action='store_true',
            help='Try to match receipt to existing transaction'
        )
    
    def handle(self, *args, **options):
        image_path = options.get('image')
        generate_sample = options.get('generate_sample')
        generate_all = options.get('generate_all')
        receipt_type = options.get('receipt_type', 'grocery')
        match_transaction = options.get('match_transaction')
        
        self.stdout.write("🧾 Starting TrOCR Receipt Parsing...")
        
        # Generate all sample receipts if requested
        if generate_all:
            receipt_types = ['grocery', 'coffee', 'restaurant', 'pharmacy']
            for rtype in receipt_types:
                image_path = self.generate_sample_receipt(rtype)
                self.stdout.write(f"✅ Generated {rtype} receipt: {image_path}")
            self.stdout.write("\n🔍 Processing all generated receipts...")
            
            # Process each generated receipt
            for rtype in receipt_types:
                self.stdout.write(f"\n{'='*50}")
                self.stdout.write(f"Processing {rtype.upper()} receipt")
                self.stdout.write('='*50)
                self.process_and_display_receipt(f'sample_receipt_{rtype}.png', match_transaction)
            return
        
        # Generate single sample receipt if requested
        if generate_sample:
            image_path = self.generate_sample_receipt(receipt_type)
            self.stdout.write(f"✅ Generated {receipt_type} receipt: {image_path}")
        
        if not image_path:
            self.stdout.write(self.style.ERROR(
                "Please provide an image path with --image or use --generate-sample"
            ))
            return
        
        # Process single receipt
        self.process_and_display_receipt(image_path, match_transaction)
    
    def process_and_display_receipt(self, image_path: str, match_transaction: bool = False):
        """Process and display results for a single receipt"""
        
        # Check if file exists
        if not os.path.exists(image_path):
            self.stdout.write(self.style.ERROR(f"Image file not found: {image_path}"))
            return
        
        # Initialize service
        service = get_receipt_service()
        
        # Process receipt
        self.stdout.write(f"\n📸 Processing receipt image: {image_path}")
        self.stdout.write("🤖 Loading TrOCR model (this may take a moment)...")
        
        result = service.process_receipt(image_path)
        
        if result['success']:
            self.stdout.write(self.style.SUCCESS("\n✅ Receipt parsed successfully!"))
            
            data = result['data']
            
            # Display extracted information
            self.stdout.write("\n📋 Extracted Information:")
            self.stdout.write(f"   Merchant: {data['merchant'] or 'Not found'}")
            self.stdout.write(f"   Date: {data['date'] or 'Not found'}")
            self.stdout.write(f"   Time: {data['time'] or 'Not found'}")
            self.stdout.write(f"   Total: ${data['total']:.2f}" if data['total'] else "   Total: Not found")
            self.stdout.write(f"   Tax: ${data['tax']:.2f}" if data['tax'] else "   Tax: Not found")
            self.stdout.write(f"   Phone: {data['phone'] or 'Not found'}")
            
            if data['items']:
                self.stdout.write(f"\n   Items ({len(data['items'])}):")
                for item in data['items'][:5]:  # Show first 5 items
                    self.stdout.write(f"     • {item['name']}: ${item['price']:.2f}")
            
            self.stdout.write(f"\n   Confidence: {data['confidence']:.1%}")
            self.stdout.write(f"\n📝 Summary: {result['summary']}")
            
            # Show raw OCR text (first 200 chars)
            if data['raw_text']:
                self.stdout.write(f"\n🔍 Raw OCR Text (preview):")
                preview = data['raw_text'][:200].replace('\n', ' ')
                self.stdout.write(f"   {preview}...")
            
            # Try to match to transaction
            if match_transaction:
                self.stdout.write("\n🔗 Attempting to match with transaction...")
                
                user = User.objects.get(username='demo_user')
                account = user.accounts.first()
                
                if account and data['total']:
                    match = service.match_receipt_to_transaction(data, account.id)
                    
                    if match:
                        self.stdout.write(self.style.SUCCESS("   ✅ Found matching transaction!"))
                        self.stdout.write(f"   Transaction: {match['description']}")
                        self.stdout.write(f"   Amount: ${abs(match['amount']):.2f}")
                        self.stdout.write(f"   Date: {match['date'][:10]}")
                        self.stdout.write(f"   Match confidence: {match['confidence']:.1%}")
                    else:
                        self.stdout.write("   ❌ No matching transaction found")
            
            # Save to JSON
            output_file = image_path.replace('.png', '_parsed.json').replace('.jpg', '_parsed.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            self.stdout.write(f"\n💾 Full results saved to: {output_file}")
            
        else:
            self.stdout.write(self.style.ERROR(f"\n❌ Receipt parsing failed: {result['error']}"))
    
    def generate_sample_receipt(self, receipt_type='grocery') -> str:
        """Generate a sample receipt image for testing
        
        Args:
            receipt_type: Type of receipt to generate ('grocery', 'coffee', 'restaurant', 'pharmacy')
        """
        
        # Create a simple receipt image with better contrast
        width, height = 400, 600
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a better font, fallback to default
        try:
            # Try different font paths for different OS
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            ]
            font_large = None
            font_normal = None
            for path in font_paths:
                if os.path.exists(path):
                    font_large = ImageFont.truetype(path, 24)
                    font_normal = ImageFont.truetype(path, 16)
                    font_small = ImageFont.truetype(path, 12)
                    break
            
            if not font_large:
                # Use default if no system fonts found
                font_large = ImageFont.load_default()
                font_normal = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_normal = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Different receipt contents based on type
        if receipt_type == 'coffee':
            receipt_data = self._get_coffee_receipt_data()
        elif receipt_type == 'restaurant':
            receipt_data = self._get_restaurant_receipt_data()
        elif receipt_type == 'pharmacy':
            receipt_data = self._get_pharmacy_receipt_data()
        else:  # default to grocery
            receipt_data = self._get_grocery_receipt_data()
        
        # Draw receipt using the data
        y = 20
        
        # Header
        draw.text((width//2 - len(receipt_data['merchant'])*6, y), receipt_data['merchant'], fill='black', font=font_large)
        y += 40
        draw.text((width//2 - len(receipt_data['subtitle'])*5, y), receipt_data['subtitle'], fill='gray', font=font_small)
        y += 30
        
        # Store info
        for info_line in receipt_data['store_info']:
            draw.text((50, y), info_line, fill='black', font=font_small)
            y += 20
        y += 20
        
        # Date and time
        draw.text((50, y), f"Date: {receipt_data['date']}", fill='black', font=font_normal)
        draw.text((250, y), f"Time: {receipt_data['time']}", fill='black', font=font_normal)
        y += 30
        
        draw.line([(30, y), (width-30, y)], fill='black', width=1)
        y += 20
        
        # Items
        for item, price in receipt_data['items']:
            draw.text((50, y), item, fill='black', font=font_normal)
            draw.text((width - 80, y), f"${price}", fill='black', font=font_normal)
            y += 25
        
        y += 10
        draw.line([(30, y), (width-30, y)], fill='black', width=1)
        y += 20
        
        # Totals
        draw.text((50, y), "Subtotal:", fill='black', font=font_normal)
        draw.text((width - 80, y), f"${receipt_data['subtotal']}", fill='black', font=font_normal)
        y += 25
        
        draw.text((50, y), f"Tax ({receipt_data['tax_rate']}%):", fill='black', font=font_normal)
        draw.text((width - 80, y), f"${receipt_data['tax']}", fill='black', font=font_normal)
        y += 25
        
        draw.text((50, y), "TOTAL:", fill='black', font=font_large)
        draw.text((width - 90, y), f"${receipt_data['total']}", fill='black', font=font_large)
        y += 40
        
        draw.line([(30, y), (width-30, y)], fill='black', width=1)
        y += 20
        
        # Footer
        draw.text((width//2 - 60, y), "THANK YOU!", fill='black', font=font_normal)
        y += 25
        draw.text((width//2 - 100, y), receipt_data['footer'], fill='gray', font=font_small)
        
        # Save image with type-specific name
        output_path = f'sample_receipt_{receipt_type}.png'
        image.save(output_path)
        
        return output_path
    
    def _get_grocery_receipt_data(self):
        """Get data for a grocery store receipt"""
        return {
            'merchant': 'WHOLE FOODS',
            'subtitle': 'Natural & Organic',
            'store_info': [
                'Store #123',
                '123 Main Street',
                'San Francisco, CA 94102',
                'Phone: (415) 555-0123'
            ],
            'date': '09/21/2025',
            'time': '14:32',
            'items': [
                ("Organic Bananas", "3.99"),
                ("Almond Milk", "4.49"),
                ("Whole Wheat Bread", "3.29"),
                ("Free Range Eggs", "5.99"),
                ("Avocados (3)", "4.50"),
                ("Greek Yogurt", "6.99"),
                ("Spinach", "2.99"),
                ("Salmon Fillet", "12.99")
            ],
            'subtotal': '45.24',
            'tax_rate': '8.75',
            'tax': '3.96',
            'total': '49.20',
            'footer': 'Save your receipt for returns'
        }
    
    def _get_coffee_receipt_data(self):
        """Get data for a coffee shop receipt"""
        return {
            'merchant': 'STARBUCKS',
            'subtitle': 'Coffee Company',
            'store_info': [
                'Store #456',
                '789 Market Street',
                'San Francisco, CA 94103',
                '(415) 555-9876'
            ],
            'date': '09/21/2025',
            'time': '08:45',
            'items': [
                ("Venti Latte", "5.95"),
                ("Blueberry Muffin", "3.50"),
                ("Extra Shot", "0.75"),
                ("Oat Milk", "0.70")
            ],
            'subtotal': '10.90',
            'tax_rate': '8.75',
            'tax': '0.95',
            'total': '11.85',
            'footer': 'Join our rewards program!'
        }
    
    def _get_restaurant_receipt_data(self):
        """Get data for a restaurant receipt"""
        return {
            'merchant': 'BLUE MOON CAFE',
            'subtitle': 'Fine Dining',
            'store_info': [
                'Table 12',
                '555 Valencia St',
                'San Francisco, CA 94110',
                'Tel: (415) 555-3333'
            ],
            'date': '09/20/2025',
            'time': '19:30',
            'items': [
                ("Caesar Salad", "12.50"),
                ("Grilled Salmon", "28.95"),
                ("House Wine", "8.50"),
                ("Tiramisu", "9.95"),
                ("Cappuccino", "4.50")
            ],
            'subtotal': '64.40',
            'tax_rate': '8.75',
            'tax': '5.64',
            'total': '70.04',
            'footer': 'Gratuity not included'
        }
    
    def _get_pharmacy_receipt_data(self):
        """Get data for a pharmacy receipt"""
        return {
            'merchant': 'WALGREENS',
            'subtitle': 'Pharmacy',
            'store_info': [
                'Store #789',
                '321 Mission St',
                'San Francisco, CA 94105',
                'Ph: (415) 555-7777'
            ],
            'date': '09/21/2025',
            'time': '11:15',
            'items': [
                ("Vitamin D3", "12.99"),
                ("Ibuprofen 200mg", "8.49"),
                ("Band-Aids", "5.99"),
                ("Hand Sanitizer", "4.99"),
                ("Tissues 3pk", "7.49")
            ],
            'subtotal': '39.95',
            'tax_rate': '8.75',
            'tax': '3.50',
            'total': '43.45',
            'footer': 'Be well!'
        }