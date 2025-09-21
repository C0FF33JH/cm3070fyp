"""
SetFit Transaction Categorization Service
Handles loading and inference for transaction categorization
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
from setfit import SetFitModel
from django.conf import settings
import torch
import numpy as np

logger = logging.getLogger(__name__)

class SetFitCategorizationService:
    """Service for categorizing transactions using SetFit model"""
    
    # Map model output labels to database categories
    # If model outputs integers, map them to categories
    LABEL_TO_CATEGORY = {
        0: 'FOOD_AND_DRINK',
        1: 'GENERAL_MERCHANDISE',
        2: 'TRANSPORTATION',
        3: 'ENTERTAINMENT',
        4: 'GENERAL_SERVICES',
        5: 'MEDICAL',
        6: 'PERSONAL_CARE',
        7: 'HOME_IMPROVEMENT',
        8: 'RENT_AND_UTILITIES',
        9: 'TRAVEL',
        10: 'LOAN_PAYMENTS',
        11: 'GOVERNMENT_AND_NON_PROFIT',
        12: 'INCOME',
        13: 'TRANSFER_IN',
        14: 'TRANSFER_OUT',
        15: 'BANK_FEES'
    }
    
    # Also support string labels
    CATEGORY_MAP = {
        'FOOD_AND_DRINK': 'FOOD_AND_DRINK',
        'GENERAL_MERCHANDISE': 'GENERAL_MERCHANDISE', 
        'TRANSPORTATION': 'TRANSPORTATION',
        'ENTERTAINMENT': 'ENTERTAINMENT',
        'GENERAL_SERVICES': 'GENERAL_SERVICES',
        'MEDICAL': 'MEDICAL',
        'PERSONAL_CARE': 'PERSONAL_CARE',
        'HOME_IMPROVEMENT': 'HOME_IMPROVEMENT',
        'RENT_AND_UTILITIES': 'RENT_AND_UTILITIES',
        'TRAVEL': 'TRAVEL',
        'LOAN_PAYMENTS': 'LOAN_PAYMENTS',
        'GOVERNMENT_AND_NON_PROFIT': 'GOVERNMENT_AND_NON_PROFIT',
        'INCOME': 'INCOME',
        'TRANSFER_IN': 'TRANSFER_IN',
        'TRANSFER_OUT': 'TRANSFER_OUT',
        'BANK_FEES': 'BANK_FEES'
    }
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(settings.MODEL_DIR, 'setfit_model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self) -> bool:
        """Load the SetFit model from disk"""
        try:
            if self.model is None:
                logger.info(f"Loading SetFit model from {self.model_path}")
                self.model = SetFitModel.from_pretrained(self.model_path)
                self.model.to(self.device)
                logger.info(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SetFit model: {e}")
            return False
    
    def prepare_text(self, description: str, merchant: Optional[str] = None) -> str:
        """
        Prepare transaction text for model input
        Combines description and merchant name as per training
        """
        text = description.strip()
        if merchant and merchant.strip():
            text = f"{text} {merchant.strip()}"
        return text
    
    def predict_single(self, description: str, merchant: Optional[str] = None) -> Tuple[str, float]:
        """
        Predict category for a single transaction
        Returns: (category, confidence_score)
        """
        if not self.load_model():
            logger.error("Model not loaded")
            return None, 0.0
        
        try:
            # Prepare input text
            text = self.prepare_text(description, merchant)
            
            # Get prediction
            predictions = self.model.predict([text])
            
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba([text])
                confidence = float(probabilities.max())
            except:
                # If predict_proba not available, use high confidence for predictions
                confidence = 0.95
            
            # Get the predicted label
            predicted_label = predictions[0]
            
            # Map to our category enum
            # Handle both integer and string labels
            if isinstance(predicted_label, (int, np.integer)):
                category = self.LABEL_TO_CATEGORY.get(predicted_label, f"UNKNOWN_{predicted_label}")
            else:
                category = self.CATEGORY_MAP.get(predicted_label, predicted_label)
            
            logger.debug(f"Predicted {category} with confidence {confidence:.2f} for: {text[:50]}...")
            
            return category, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, 0.0
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Predict categories for multiple transactions
        
        Args:
            transactions: List of dicts with 'description' and optional 'merchant' keys
            
        Returns:
            List of dicts with 'category' and 'confidence' keys added
        """
        if not self.load_model():
            logger.error("Model not loaded")
            return transactions
        
        try:
            # Prepare all texts
            texts = [
                self.prepare_text(t['description'], t.get('merchant'))
                for t in transactions
            ]
            
            # Batch prediction
            predictions = self.model.predict(texts)
            
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba(texts)
                confidences = probabilities.max(axis=1)
            except:
                # Default high confidence if predict_proba not available
                confidences = [0.95] * len(texts)
            
            # Add predictions to transactions
            for i, transaction in enumerate(transactions):
                predicted_label = predictions[i]
                
                # Handle both integer and string labels
                if isinstance(predicted_label, (int, np.integer)):
                    category = self.LABEL_TO_CATEGORY.get(predicted_label, f"UNKNOWN_{predicted_label}")
                else:
                    category = self.CATEGORY_MAP.get(predicted_label, predicted_label)
                
                transaction['ai_category'] = category
                transaction['confidence_score'] = float(confidences[i])
                
            logger.info(f"Batch prediction completed for {len(transactions)} transactions")
            
            return transactions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return transactions
    
    def categorize_uncategorized_transactions(self, account_id: Optional[int] = None):
        """
        Find and categorize all uncategorized transactions in the database
        
        Args:
            account_id: Optional account ID to filter transactions
        """
        from dashboard.models import Transaction
        
        # Query uncategorized transactions
        query = Transaction.objects.filter(
            ai_category__isnull=True,
            data_source='plaid'  # Only categorize open banking data
        )
        
        if account_id:
            query = query.filter(account_id=account_id)
        
        transactions = query.values('id', 'description', 'merchant')
        
        if not transactions:
            logger.info("No uncategorized transactions found")
            return 0
        
        logger.info(f"Found {len(transactions)} uncategorized transactions")
        
        # Batch process
        batch_size = 32
        total_processed = 0
        
        for i in range(0, len(transactions), batch_size):
            batch = list(transactions[i:i+batch_size])
            
            # Predict categories
            results = self.predict_batch(batch)
            
            # Update database
            for result in results:
                if result.get('ai_category'):
                    Transaction.objects.filter(id=result['id']).update(
                        ai_category=result['ai_category'],
                        confidence_score=result['confidence_score']
                    )
                    total_processed += 1
        
        logger.info(f"Categorized {total_processed} transactions")
        return total_processed
    
    def get_category_statistics(self, account_id: Optional[int] = None) -> Dict:
        """
        Get statistics about categorized transactions
        """
        from dashboard.models import Transaction
        from django.db.models import Count, Sum, Avg
        
        query = Transaction.objects.filter(
            ai_category__isnull=False,
            data_source='plaid'
        )
        
        if account_id:
            query = query.filter(account_id=account_id)
        
        stats = query.values('ai_category').annotate(
            count=Count('id'),
            total_amount=Sum('amount'),
            avg_amount=Avg('amount'),
            avg_confidence=Avg('confidence_score')
        ).order_by('-total_amount')
        
        return {
            'categories': list(stats),
            'total_transactions': query.count(),
            'average_confidence': query.aggregate(Avg('confidence_score'))['confidence_score__avg']
        }


# Singleton instance
_categorization_service = None

def get_categorization_service() -> SetFitCategorizationService:
    """Get or create the singleton categorization service"""
    global _categorization_service
    if _categorization_service is None:
        _categorization_service = SetFitCategorizationService()
    return _categorization_service