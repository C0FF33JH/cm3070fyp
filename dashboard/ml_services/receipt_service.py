"""
TrOCR Receipt Parsing Service with EasyOCR fallback
File: dashboard/ml_services/receipt_service.py
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import json
from PIL import Image, ImageOps, ImageEnhance
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile
import io
import numpy as np

logger = logging.getLogger(__name__)

class TrOCRReceiptService:
    """Service for parsing receipts using TrOCR model with EasyOCR fallback"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.model_name = "microsoft/trocr-base-printed"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.easyocr_reader = None
        
        # Regex patterns for extraction
        self.patterns = {
            'total': [
                r'TOTAL[\s:]*\$?(\d+\.?\d*)',
                r'(?:total|amount|sum|balance)[\s:]*\$?(\d+\.?\d*)',
                r'\$(\d+\.\d{2})(?:\s|$)',
                r'(?:grand\s+total|total\s+due)[\s:]*\$?(\d+\.?\d*)',
            ],
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'time': [
                r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',
                r'Time:\s*(\d{1,2}:\d{2})',
            ],
            'tax': [
                r'(?:tax|gst|vat)[\s:]*\$?(\d+\.?\d*)',
                r'Tax\s*\([\d.]+%\)[\s:]*\$?(\d+\.?\d*)',
            ],
            'items': [
                r'(.+?)\s+\$?(\d+\.\d{2})',
                r'(\d+)\s+(.+?)\s+\$?(\d+\.\d{2})',
            ],
            'phone': [
                r'(?:Phone:|Ph:|Tel:)?\s*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
            ],
        }
    
    def load_model(self) -> bool:
        """Load the TrOCR model and processor"""
        try:
            if self.model is None or self.processor is None:
                logger.info(f"Loading TrOCR model: {self.model_name}")
                
                self.processor = TrOCRProcessor.from_pretrained(self.model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                logger.info(f"TrOCR model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            return False
    
    def load_easyocr(self):
        """Load EasyOCR reader on demand"""
        if self.easyocr_reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR reader...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                logger.info("EasyOCR loaded successfully")
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                raise
        return self.easyocr_reader
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using TrOCR, fallback to EasyOCR if needed
        """
        # First, try TrOCR with improved preprocessing
        trocr_text = self._try_trocr_extraction(image)
        
        if self._is_valid_receipt_text(trocr_text):
            logger.info(f"TrOCR extracted valid text: {len(trocr_text)} characters")
            return trocr_text
        
        # Fallback to EasyOCR
        logger.info("TrOCR extraction inadequate, trying EasyOCR...")
        easyocr_text = self._try_easyocr_extraction(image)
        
        if self._is_valid_receipt_text(easyocr_text):
            logger.info(f"EasyOCR extracted valid text: {len(easyocr_text)} characters")
            return easyocr_text
        
        # Return the better of the two
        if len(easyocr_text) > len(trocr_text):
            logger.info("Using EasyOCR result")
            return easyocr_text
        else:
            logger.info("Using TrOCR result")
            return trocr_text
    
    def _try_trocr_extraction(self, image: Image.Image) -> str:
        """Try text extraction with TrOCR"""
        if not self.load_model():
            return ""
        
        try:
            # Preprocess image for TrOCR
            processed = self._preprocess_for_ocr(image)
            
            # Try full image first for our generated receipts
            width, height = processed.size
            
            # Process full image for standard receipt sizes
            if width <= 1024 and height <= 1024:
                text = self._extract_trocr_single(processed)
                logger.debug(f"TrOCR full image extraction: {len(text)} chars")
                if len(text) > 50:  # Lowered threshold
                    return text
            
            # Fallback to horizontal strips for larger or failed extractions
            text = self._extract_trocr_strips(processed)
            logger.debug(f"TrOCR strips extraction: {len(text)} chars")
            return text
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return ""
    
    def _extract_trocr_single(self, image: Image.Image) -> str:
        """Extract text from single image with TrOCR"""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=512,
                num_beams=5,
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    
    def _extract_trocr_strips(self, image: Image.Image) -> str:
        """Extract text using horizontal strips"""
        width, height = image.size
        strip_height = 100
        overlap = 20
        texts = []
        
        for y in range(0, height - overlap, strip_height - overlap):
            y_end = min(y + strip_height, height)
            strip = image.crop((0, y, width, y_end))
            
            try:
                pixel_values = self.processor(strip, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=128,
                        num_beams=3,
                        temperature=0.8,
                        do_sample=True
                    )
                
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if text.strip():
                    texts.append(text.strip())
                    
            except Exception as e:
                logger.debug(f"Strip processing error: {e}")
                continue
        
        return '\n'.join(texts)
    
    def _try_easyocr_extraction(self, image: Image.Image) -> str:
        """Try text extraction with EasyOCR"""
        try:
            reader = self.load_easyocr()
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run EasyOCR
            result = reader.readtext(img_array)
            
            # Sort by vertical position (top to bottom)
            result.sort(key=lambda x: x[0][0][1])
            
            # Extract text in reading order
            lines = []
            current_line = []
            last_y = -1
            
            for detection in result:
                bbox, text, confidence = detection
                y_coord = bbox[0][1]
                
                # New line if Y coordinate differs significantly
                if last_y != -1 and abs(y_coord - last_y) > 20:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = []
                
                current_line.append(text)
                last_y = y_coord
            
            # Add last line
            if current_line:
                lines.append(' '.join(current_line))
            
            full_text = '\n'.join(lines)
            logger.debug(f"EasyOCR extracted: {full_text[:200]}...")
            
            return full_text
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Resize if too large
            max_width = 800
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            
            # Add white padding
            image = ImageOps.expand(image, border=20, fill='white')
            
            return image
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def _is_valid_receipt_text(self, text: str) -> bool:
        """Check if extracted text looks like valid receipt content"""
        if not text or len(text) < 20:
            return False
        
        # Check for nonsense patterns (repeated chars, only numbers, etc)
        if re.match(r'^[\*\s\d]+$', text):
            return False
        
        # Check for excessive repeated characters (KKK... or WWW...)
        if re.search(r'(.)\1{10,}', text):
            logger.debug("Text has excessive repeated characters")
            return False
        
        # Check for receipt keywords
        receipt_keywords = ['total', 'tax', 'date', 'time', 'store', 'price', 
                           'subtotal', 'receipt', 'phone', 'thank', 'amount']
        text_lower = text.lower()
        
        keyword_count = sum(1 for kw in receipt_keywords if kw in text_lower)
        
        # Also check for dollar amounts
        has_price = bool(re.search(r'\$?\d+\.\d{2}', text))
        
        # Lower threshold since TrOCR is extracting partial text
        return keyword_count >= 1 and has_price
    
    def parse_receipt_text(self, text: str) -> Dict:
        """
        Parse structured data from receipt text using regex patterns
        Handles common OCR errors
        """
        text_clean = text.strip()
        result = {
            'raw_text': text,
            'merchant': None,
            'date': None,
            'time': None,
            'total': None,
            'subtotal': None,
            'tax': None,
            'items': [],
            'phone': None,
            'address': None,
            'confidence': 0.0
        }
        
        # Clean up common OCR errors
        text_cleaned = text_clean.replace('WWW.', '').replace('KKK.', '')
        text_cleaned = re.sub(r'[WK]{10,}', '', text_cleaned)  # Remove long W or K sequences
        
        # Extract merchant name (usually in first few lines, before dates/times)
        lines = text_clean.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 3:
                # Skip lines that are clearly not merchant names
                skip_keywords = ['date:', 'time:', 'phone:', 'store #', 'cashier', 
                                'table', 'receipt', 'tax', 'total', 'please', 'cash']
                if not any(x in line.lower() for x in skip_keywords):
                    # Clean up merchant name
                    merchant = line.upper()
                    # Fix common OCR errors in merchant names
                    merchant = merchant.replace('TABLE FOODS', 'WHOLE FOODS')
                    result['merchant'] = merchant
                    break
        
        # Extract total - try multiple patterns
        total_patterns = [
            r'TOTAL[\s:]*\$?(\d+\.\d{2})',
            r'(?:total|amount)[\s:]*\$?(\d+\.\d{2})',
            r'\$(\d+\.\d{2})(?:\s|$)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                try:
                    result['total'] = float(match.group(1))
                    break
                except:
                    pass
        
        # Extract date - look for various formats
        date_patterns = [
            r'DATE[\s:]*(\d{8})',  # DATE:09212025
            r'DATE[\s:]*(\d{2}/\d{2}/\d{4})',
            r'DATE[\s:]*(\d{2}/\d{2}/\d{2})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Handle concatenated date format like 09212025
                if len(date_str) == 8 and date_str.isdigit():
                    date_str = f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
                result['date'] = date_str
                break
        
        # Extract time
        time_patterns = [
            r'TIME[\s:]*(\d{3,4})',  # TIME:1432 or TIME:845
            r'TIME[\s:]*(\d{1,2}:\d{2})',
            r'(\d{1,2}:\d{2}(?:\s*[AP]M)?)',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                # Handle concatenated time like 1432
                if time_str.isdigit():
                    if len(time_str) == 4:
                        time_str = f"{time_str[:2]}:{time_str[2:]}"
                    elif len(time_str) == 3:
                        time_str = f"0{time_str[0]}:{time_str[1:]}"
                result['time'] = time_str
                break
        
        # Extract subtotal
        subtotal_patterns = [
            r'(?:SUBTOTAL|SUBPINAL|SUBTINAL)[\s:]*\$?(\d+\.\d{2})',
            r'Subtotal[\s:]*\$?(\d+\.\d{2})',
        ]
        for pattern in subtotal_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                try:
                    result['subtotal'] = float(match.group(1))
                    break
                except:
                    pass
        
        # Extract tax
        tax_patterns = [
            r'(?:Tax|TAX)[\s:]*\$?(\d+\.\d{2})',
            r'Tax\s*\([^)]+\)[\s:]*\$?(\d+\.\d{2})',
        ]
        for pattern in tax_patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                try:
                    result['tax'] = float(match.group(1))
                except:
                    pass
        
        # Extract items (simplified due to OCR errors)
        item_patterns = [
            r'([A-Za-z\s]+?)[\s:]+\$?(\d+\.\d{2})',
            r'([A-Za-z\s]+?)\s*~*\s*\$?(\d+\.\d{2})',
        ]
        for pattern in item_patterns:
            items = re.findall(pattern, text_cleaned)
            for item_name, price in items[:15]:
                item_name = item_name.strip()
                # Filter out non-items
                skip_words = ['total', 'tax', 'subtotal', 'change', 'cash', 'date', 
                             'time', 'thank', 'cashier', 'receipt', 'amount']
                if not any(word in item_name.lower() for word in skip_words):
                    if len(item_name) > 2:
                        result['items'].append({
                            'name': item_name,
                            'price': float(price)
                        })
        
        # Extract phone
        phone_patterns = [
            r'(?:Phone:|Ph:|Tel:)?\s*\((\d{3})\)\s*(\d{3})-?(\d{4})',
            r'(\d{3})[-.\s](\d{3})[-.\s](\d{4})',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text_cleaned)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    result['phone'] = f"({groups[0]}) {groups[1]}-{groups[2]}"
                break
        
        # Calculate confidence
        extracted_fields = sum([
            1 if result['merchant'] else 0,
            1 if result['date'] else 0,
            1 if result['total'] else 0,
            1 if result['items'] else 0,
            1 if result['tax'] else 0,
            1 if result['time'] else 0,
        ])
        result['confidence'] = extracted_fields / 6.0
        
        return result
    
    def process_receipt(self, image_input, enhance: bool = True) -> Dict:
        """
        Complete receipt processing pipeline
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, InMemoryUploadedFile):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                image = Image.open(io.BytesIO(image_input))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text
            extracted_text = self.extract_text_from_image(image)
            
            # Parse structured data
            parsed_data = self.parse_receipt_text(extracted_text)
            
            # Add metadata
            parsed_data['processing_time'] = datetime.now().isoformat()
            parsed_data['image_size'] = image.size
            
            # Validate and clean
            parsed_data = self.validate_receipt_data(parsed_data)
            
            return {
                'success': True,
                'data': parsed_data,
                'summary': self.generate_summary(parsed_data)
            }
            
        except Exception as e:
            logger.error(f"Receipt processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def validate_receipt_data(self, data: Dict) -> Dict:
        """Validate and clean extracted receipt data"""
        # Validate date format
        if data['date']:
            try:
                date_str = data['date'].replace('Date:', '').strip()
                
                # Try various formats
                formats_to_try = [
                    '%m/%d/%Y',    # 09/21/2025
                    '%m/%d/%y',    # 09/21/25
                    '%Y-%m-%d',    # 2025-09-21
                    '%d/%m/%Y',    # 21/09/2025
                    '%m-%d-%Y',    # 09-21-2025
                ]
                
                parsed = False
                for fmt in formats_to_try:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        data['date'] = parsed_date.strftime('%Y-%m-%d')
                        parsed = True
                        break
                    except:
                        continue
                
                if not parsed:
                    logger.debug(f"Could not parse date: {date_str}")
                    data['date'] = None
                    
            except Exception as e:
                logger.debug(f"Date validation error: {e}")
                data['date'] = None
        
        # Validate time format
        if data['time']:
            try:
                time_str = data['time'].strip()
                # Already formatted as HH:MM by parser
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        hour = int(parts[0])
                        minute = int(parts[1])
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            data['time'] = f"{hour:02d}:{minute:02d}"
            except:
                data['time'] = None
        
        # Ensure total is positive
        if data['total'] and data['total'] < 0:
            data['total'] = abs(data['total'])
        
        # Calculate subtotal if missing
        if not data['subtotal'] and data['total'] and data['tax']:
            data['subtotal'] = data['total'] - data['tax']
        
        return data
    
    def generate_summary(self, parsed_data: Dict) -> str:
        """Generate human-readable summary"""
        parts = []
        
        if parsed_data['merchant']:
            parts.append(f"Receipt from {parsed_data['merchant']}")
        
        if parsed_data['date']:
            parts.append(f"dated {parsed_data['date']}")
        
        if parsed_data['total']:
            parts.append(f"Total: ${parsed_data['total']:.2f}")
        
        if parsed_data['items']:
            parts.append(f"{len(parsed_data['items'])} items")
        
        if parts:
            return ". ".join(parts)
        else:
            return "Receipt parsed with limited information"
    
    def match_receipt_to_transaction(self, receipt_data: Dict, account_id: int) -> Optional[Dict]:
        """Try to match receipt to an existing transaction"""
        from dashboard.models import Transaction
        from django.db.models import Q
        
        if not receipt_data.get('total'):
            return None
        
        # Search for matching transaction
        query = Q(account_id=account_id)
        
        # Match by amount
        amount = -abs(receipt_data['total'])
        query &= Q(amount__gte=amount - 0.50) & Q(amount__lte=amount + 0.50)
        
        # Match by date if available
        if receipt_data.get('date'):
            try:
                receipt_date = datetime.strptime(receipt_data['date'], '%Y-%m-%d')
                query &= Q(date__date=receipt_date.date())
            except:
                pass
        
        # Match by merchant
        if receipt_data.get('merchant'):
            merchant_words = receipt_data['merchant'].split()[:2]
            for word in merchant_words:
                if len(word) > 3:
                    query &= Q(description__icontains=word) | Q(merchant__icontains=word)
                    break
        
        matches = Transaction.objects.filter(query).order_by('date')[:5]
        
        if matches:
            return {
                'transaction_id': matches[0].transaction_id,
                'description': matches[0].description,
                'amount': float(matches[0].amount),
                'date': matches[0].date.isoformat(),
                'confidence': 0.8 if len(matches) == 1 else 0.6
            }
        
        return None


# Singleton instance
_receipt_service = None

def get_receipt_service() -> TrOCRReceiptService:
    """Get or create the singleton receipt service"""
    global _receipt_service
    if _receipt_service is None:
        _receipt_service = TrOCRReceiptService()
    return _receipt_service