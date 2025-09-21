"""
XGBoost Fraud Detection Service
Handles fraud/anomaly detection for transactions with enhanced features
File: dashboard/ml_services/fraud_service.py
"""

import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

class XGBoostFraudService:
    """Service for detecting fraudulent transactions using XGBoost model"""
    
    # Feature columns required by the model (based on your trained model)
    REQUIRED_FEATURES = [
        'amt', 'amt_log', 'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_night', 'age', 'distance_km', 'distance_from_home',
        'category_freq', 'merchant_freq', 'city_pop', 'city_pop_log',
        'lat', 'long', 'merch_lat', 'merch_long', 'high_risk_time',
        'amt_percentile_by_category', 'category_encoded', 'gender_encoded',
        'state_encoded'
    ]
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(settings.MODEL_DIR, 'xgboost_model.pkl')
        self.threshold = 0.5  # Default fraud threshold
        
    def load_model(self) -> bool:
        """Load the XGBoost model from disk"""
        try:
            if self.model is None:
                logger.info(f"Loading XGBoost model from {self.model_path}")
                
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    
                    with open(self.model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Handle different pickle formats
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.threshold = model_data.get('threshold', 0.5)
                        self.feature_names = model_data.get('feature_names', self.REQUIRED_FEATURES)
                    else:
                        self.model = model_data
                        self.feature_names = self.REQUIRED_FEATURES
                
                logger.info(f"XGBoost model loaded successfully")
                logger.info(f"Fraud threshold: {self.threshold}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False
    
    def prepare_features(self, transaction_data: Dict) -> pd.DataFrame:
        """
        Prepare features for XGBoost model from transaction data
        Matching exact features from training
        """
        features = {}
        
        # Basic amount features
        amt = abs(float(transaction_data.get('amount', 0)))
        features['amt'] = amt
        features['amt_log'] = np.log1p(amt)  # log(1 + amt) to handle 0
        
        # Geographic features
        features['lat'] = float(transaction_data.get('lat', 0))
        features['long'] = float(transaction_data.get('long', 0))
        features['merch_lat'] = float(transaction_data.get('merch_lat', 0))
        features['merch_long'] = float(transaction_data.get('merch_long', 0))
        
        # City population features
        city_pop = int(transaction_data.get('city_pop', 0))
        features['city_pop'] = city_pop
        features['city_pop_log'] = np.log1p(city_pop)
        
        # Calculate age from DOB
        if transaction_data.get('dob'):
            dob = pd.to_datetime(transaction_data['dob'])
            trans_date = pd.to_datetime(transaction_data.get('date', timezone.now()))
            
            # Make both timezone-naive for comparison
            if trans_date.tzinfo is not None:
                trans_date = trans_date.replace(tzinfo=None)
            if dob.tzinfo is not None:
                dob = dob.replace(tzinfo=None)
                
            age = (trans_date - dob).days / 365.25
            features['age'] = age
        else:
            features['age'] = 35  # Default age
        
        # Time-based features
        trans_date = pd.to_datetime(transaction_data.get('date', timezone.now()))
        if trans_date.tzinfo is not None:
            trans_date = trans_date.replace(tzinfo=None)
            
        features['hour'] = trans_date.hour
        features['day_of_week'] = trans_date.dayofweek
        features['day_of_month'] = trans_date.day
        features['month'] = trans_date.month
        features['is_weekend'] = int(trans_date.dayofweek >= 5)
        features['is_night'] = int(trans_date.hour < 6 or trans_date.hour >= 22)
        features['high_risk_time'] = int(trans_date.hour in [0, 1, 2, 3, 4, 5])  # Early morning
        
        # Distance calculations
        features['distance_km'] = self._calculate_distance(
            features['lat'], features['long'],
            features['merch_lat'], features['merch_long']
        )
        
        # Distance from home (for demo, use a fixed home location or same as transaction)
        # In real implementation, this would be user's home location
        features['distance_from_home'] = features['distance_km']  # Simplified
        
        # Frequency features (would need historical data in production)
        # For now, using placeholder values
        features['category_freq'] = 10  # Average frequency for category
        features['merchant_freq'] = 5   # Average frequency for merchant
        features['amt_percentile_by_category'] = 50  # Median percentile
        
        # Encoded categorical features
        # These would normally come from LabelEncoders fitted during training
        features['category_encoded'] = self._encode_category(transaction_data.get('category', 'misc'))
        features['gender_encoded'] = 1 if transaction_data.get('gender', 'M') == 'M' else 0
        features['state_encoded'] = self._encode_state(transaction_data.get('state', 'CA'))
        
        # Ensure all required features are present
        for feature in self.REQUIRED_FEATURES:
            if feature not in features:
                features[feature] = 0
                logger.warning(f"Missing feature {feature}, using default value 0")
        
        return pd.DataFrame([features])[self.REQUIRED_FEATURES]
    
    def _encode_category(self, category: str) -> int:
        """Simple category encoding (would use fitted LabelEncoder in production)"""
        categories = {
            'grocery_pos': 0, 'gas_transport': 1, 'misc_net': 2, 'grocery_net': 3,
            'entertainment': 4, 'food_dining': 5, 'misc_pos': 6, 'health_fitness': 7,
            'personal_care': 8, 'travel': 9, 'kids_pets': 10, 'shopping_pos': 11,
            'shopping_net': 12, 'home': 13
        }
        return categories.get(category.lower(), 6)  # Default to misc_pos
    
    def _encode_state(self, state: str) -> int:
        """Simple state encoding (would use fitted LabelEncoder in production)"""
        # Top 10 states by population
        states = {
            'CA': 0, 'TX': 1, 'FL': 2, 'NY': 3, 'PA': 4,
            'IL': 5, 'OH': 6, 'GA': 7, 'NC': 8, 'MI': 9
        }
        return states.get(state.upper(), 10)  # Others
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def predict_fraud(self, transaction: Dict) -> Tuple[bool, float, Dict]:
        """
        Predict if a transaction is fraudulent
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Tuple of (is_fraud, probability, details)
        """
        if not self.load_model():
            logger.error("Model not loaded")
            return False, 0.0, {'error': 'Model not loaded'}
        
        try:
            # Prepare features
            features_df = self.prepare_features(transaction)
            
            # Get prediction probability
            prob = self.model.predict_proba(features_df)[0, 1]
            is_fraud = prob > self.threshold
            
            # Get feature importance for this prediction
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.REQUIRED_FEATURES,
                    self.model.feature_importances_
                ))
                
                # Get top risk factors
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                risk_factors = [f[0] for f in sorted_features]
            else:
                risk_factors = []
            
            details = {
                'probability': float(prob),
                'threshold': self.threshold,
                'is_fraud': bool(is_fraud),
                'risk_factors': risk_factors,
                'risk_level': self._get_risk_level(prob)
            }
            
            logger.debug(f"Fraud prediction: {prob:.2%} probability, fraud={is_fraud}")
            
            return is_fraud, float(prob), details
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return False, 0.0, {'error': str(e)}
    
    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        if probability < 0.2:
            return 'very_low'
        elif probability < 0.4:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        elif probability < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def analyze_batch(self, account_id: Optional[int] = None) -> Dict:
        """
        Analyze all Kaggle transactions for fraud
        
        Args:
            account_id: Optional account ID to filter
            
        Returns:
            Analysis results with statistics
        """
        from dashboard.models import Transaction
        
        # Query Kaggle transactions (they have the required features)
        query = Transaction.objects.filter(
            data_source='kaggle',
            lat__isnull=False,  # Ensure we have geographic data
            merch_lat__isnull=False
        )
        
        if account_id:
            query = query.filter(account_id=account_id)
        
        transactions = query.values()
        
        if not transactions:
            logger.warning("No Kaggle transactions found for fraud analysis")
            return {'error': 'No transactions with required features found'}
        
        logger.info(f"Analyzing {len(transactions)} transactions for fraud")
        
        results = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for trans in transactions:
            # Prepare transaction data
            trans_data = {
                'amount': trans['amount'],
                'lat': trans['lat'],
                'long': trans['long'],
                'city_pop': trans['city_pop'],
                'merch_lat': trans['merch_lat'],
                'merch_long': trans['merch_long'],
                'dob': trans['dob'],
                'date': trans['date'],
                'category': trans.get('category', 'misc'),
                'gender': trans.get('gender', 'M'),
                'state': trans.get('state', 'CA')
            }
            
            # Get prediction
            is_fraud_pred, prob, details = self.predict_fraud(trans_data)
            
            # Compare with actual label
            actual_fraud = trans['is_anomaly']
            
            # Update confusion matrix
            if actual_fraud and is_fraud_pred:
                true_positives += 1
            elif not actual_fraud and is_fraud_pred:
                false_positives += 1
            elif not actual_fraud and not is_fraud_pred:
                true_negatives += 1
            else:
                false_negatives += 1
            
            # Store result
            results.append({
                'transaction_id': trans['transaction_id'],
                'amount': float(trans['amount']),
                'merchant': trans['merchant'],
                'predicted_fraud': is_fraud_pred,
                'actual_fraud': actual_fraud,
                'probability': prob,
                'risk_level': details.get('risk_level', 'unknown')
            })
            
            # Update database with prediction
            Transaction.objects.filter(id=trans['id']).update(
                anomaly_score=prob
            )
        
        # Calculate metrics
        total = len(results)
        actual_fraud_count = true_positives + false_negatives
        predicted_fraud_count = true_positives + false_positives
        
        # Avoid division by zero
        precision = true_positives / predicted_fraud_count if predicted_fraud_count > 0 else 0
        recall = true_positives / actual_fraud_count if actual_fraud_count > 0 else 0
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Sort by probability to get top risks
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'total_transactions': total,
            'actual_fraud_count': actual_fraud_count,
            'predicted_fraud_count': predicted_fraud_count,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
            'top_risks': results[:10],
            'risk_distribution': self._get_risk_distribution(results)
        }
    
    def _get_risk_distribution(self, results: List[Dict]) -> Dict:
        """Calculate risk level distribution"""
        distribution = {
            'very_low': 0,
            'low': 0,
            'medium': 0,
            'high': 0,
            'very_high': 0
        }
        
        for r in results:
            distribution[r['risk_level']] += 1
        
        return distribution
    
    def detect_anomalies_plaid(self, account_id: int, sensitivity: float = 2.0) -> Dict:
        """
        Detect anomalies in Plaid transactions using statistical methods
        (Since they don't have geographic features for XGBoost)
        
        Args:
            account_id: Account to analyze
            sensitivity: Standard deviations for anomaly threshold
            
        Returns:
            Anomaly detection results
        """
        from dashboard.models import Transaction
        from django.db.models import Avg, StdDev
        
        # Get transactions
        transactions = Transaction.objects.filter(
            account_id=account_id,
            data_source='plaid'
        ).order_by('-date')
        
        # Calculate statistics by category
        category_stats = {}
        for category in Transaction.CATEGORY_CHOICES:
            cat_code = category[0]
            stats = transactions.filter(
                ai_category=cat_code
            ).aggregate(
                avg_amount=Avg('amount'),
                std_amount=StdDev('amount')
            )
            
            if stats['avg_amount'] is not None:
                category_stats[cat_code] = stats
        
        # Find anomalies
        anomalies = []
        for trans in transactions[:100]:  # Check recent 100
            if trans.ai_category in category_stats:
                stats = category_stats[trans.ai_category]
                if stats['std_amount']:
                    z_score = abs((float(trans.amount) - stats['avg_amount']) / stats['std_amount'])
                    
                    if z_score > sensitivity:
                        anomalies.append({
                            'transaction_id': trans.transaction_id,
                            'date': trans.date.isoformat(),
                            'description': trans.description,
                            'amount': float(trans.amount),
                            'category': trans.ai_category,
                            'z_score': z_score,
                            'expected_range': (
                                stats['avg_amount'] - sensitivity * stats['std_amount'],
                                stats['avg_amount'] + sensitivity * stats['std_amount']
                            )
                        })
        
        return {
            'total_analyzed': transactions.count(),
            'anomalies_found': len(anomalies),
            'anomalies': anomalies[:10],  # Top 10
            'sensitivity': sensitivity,
            'method': 'statistical_zscore'
        }


# Singleton instance
_fraud_service = None

def get_fraud_service() -> XGBoostFraudService:
    """Get or create the singleton fraud detection service"""
    global _fraud_service
    if _fraud_service is None:
        _fraud_service = XGBoostFraudService()
    return _fraud_service