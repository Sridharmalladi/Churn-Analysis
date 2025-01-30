import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        
    def train_random_forest(self):
        """Train Random Forest model"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        self.predictions['random_forest'] = rf.predict(self.X_test)
        
    def train_xgboost(self):
        """Train XGBoost model"""
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model
        self.predictions['xgboost'] = xgb_model.predict(self.X_test)
    
    def evaluate_models(self):
        """Evaluate all models and return metrics"""
        metrics = {}
        
        for model_name, predictions in self.predictions.items():
            metrics[model_name] = {
                'accuracy': accuracy_score(self.y_test, predictions),
                'precision': precision_score(self.y_test, predictions),
                'recall': recall_score(self.y_test, predictions),
                'f1': f1_score(self.y_test, predictions)
            }
            
        return metrics
    
    def save_best_model(self, metrics):
        """Save the best performing model based on F1 score"""
        best_model_name = max(metrics.items(), 
                            key=lambda x: x[1]['f1'])[0]
        best_model = self.models[best_model_name]
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(best_model, 'models/best_model.pkl')
        return best_model_name

    def train_and_evaluate(self):
        """Train all models and evaluate them"""
        self.train_random_forest()
        self.train_xgboost()
        
        metrics = self.evaluate_models()
        best_model = self.save_best_model(metrics)
        
        return metrics, best_model

if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    
    # Train and evaluate models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    metrics, best_model = trainer.train_and_evaluate()
    
    # Print results
    print("\nModel Metrics:")
    print(pd.DataFrame(metrics))
    print(f"\nBest Model: {best_model}")