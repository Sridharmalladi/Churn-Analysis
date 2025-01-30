import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load raw data from CSV file"""
        self.raw_data = pd.read_csv(self.filepath)
        print("Raw data loaded successfully")
        return self.raw_data
    
    def clean_data(self):
        """Clean the raw data"""
        df = self.raw_data.copy()
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert categorical variables
        categorical_columns = ['subscription_tier', 'payment_method', 'country']
        df = pd.get_dummies(df, columns=categorical_columns)
        
        self.processed_data = df
        print("Data cleaned successfully")
        return self.processed_data
    
    def split_and_save_data(self, test_size=0.2, random_state=42):
        """Split data and save all versions"""
        # Create processed directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save complete processed data
        self.processed_data.to_csv('data/processed/processed_data.csv', index=False)
        print("Saved processed_data.csv")
        
        # Split the data
        X = self.processed_data.drop('churn', axis=1)
        y = self.processed_data['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Save split data
        X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
        X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
        
        print("Split data saved successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor("data/raw/customer_data.csv")
    
    # Process data
    raw_data = processor.load_data()
    processed_data = processor.clean_data()
    X_train, X_test, y_train, y_test = processor.split_and_save_data()
    
    print("Data processing completed successfully!")