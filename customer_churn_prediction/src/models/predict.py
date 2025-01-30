import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self, model_path='models/best_model.pkl'):
        """Load the trained model"""
        self.model = joblib.load(model_path)
    
    def preprocess_features(self, features):
        """Preprocess the input features"""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        # Ensure all required features are present
        required_features = [
            'account_age_days', 'days_since_last_purchase',
            'avg_transaction_value', 'monthly_spend',
            'purchase_frequency', 'support_ticket_ratio'
        ]
        
        for feature in required_features:
            if feature not in features.columns:
                features[feature] = 0
                
        return features
    
    def predict(self, features):
        """Make prediction for new data"""
        # Preprocess features
        processed_features = self.preprocess_features(features)
        
        # Make prediction
        prediction = self.model.predict(processed_features)
        probability = self.model.predict_proba(processed_features)
        
        return {
            'prediction': prediction[0],
            'churn_probability': probability[0][1]
        }
    
    def batch_predict(self, features_df):
        """Make predictions for multiple customers"""
        processed_features = self.preprocess_features(features_df)
        predictions = self.model.predict(processed_features)
        probabilities = self.model.predict_proba(processed_features)
        
        return pd.DataFrame({
            'prediction': predictions,
            'churn_probability': probabilities[:, 1]
        })

if __name__ == "__main__":
    # Example usage
    predictor = ChurnPredictor()
    
    # Single prediction example
    sample_customer = {
        'account_age_days': 365,
        'days_since_last_purchase': 30,
        'avg_transaction_value': 100,
        'monthly_spend': 300,
        'purchase_frequency': 2.5,
        'support_ticket_ratio': 0.1
    }
    
    result = predictor.predict(sample_customer)
    print("Prediction Result:", result)