import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def create_time_features(self):
        """Create features based on time-related data"""
        self.df['account_created'] = pd.to_datetime(self.df['account_created'])
        self.df['last_purchase'] = pd.to_datetime(self.df['last_purchase'])
        
        self.df['account_age_days'] = (datetime.now() - 
                                     self.df['account_created']).dt.days
        self.df['days_since_last_purchase'] = (datetime.now() - 
                                             self.df['last_purchase']).dt.days
        
        return self.df
    
    def create_monetary_features(self):
        """Create features based on monetary values"""
        self.df['avg_transaction_value'] = (self.df['total_purchase_amount'] / 
                                          self.df['number_of_purchases'])
        self.df['monthly_spend'] = (self.df['total_purchase_amount'] / 
                                  (self.df['account_age_days'] / 30))
        return self.df
    
    def create_engagement_features(self):
        """Create features based on customer engagement"""
        self.df['purchase_frequency'] = (self.df['number_of_purchases'] / 
                                       self.df['account_age_days'])
        self.df['support_ticket_ratio'] = (self.df['support_tickets'] / 
                                         self.df['number_of_purchases'])
        return self.df
    
    def scale_features(self, columns_to_scale):
        """Scale numerical features"""
        scaler = StandardScaler()
        self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
        return self.df
    
    def engineer_features(self):
        """Run complete feature engineering pipeline"""
        self.create_time_features()
        self.create_monetary_features()
        self.create_engagement_features()
        
        numerical_columns = [
            'account_age_days', 'days_since_last_purchase', 
            'avg_transaction_value', 'monthly_spend',
            'purchase_frequency', 'support_ticket_ratio'
        ]
        
        self.scale_features(numerical_columns)
        return self.df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Engineer features
    engineer = FeatureEngineer(df)
    engineered_df = engineer.engineer_features()
    
    # Save engineered features
    engineered_df.to_csv("data/processed/engineered_features.csv", index=False)