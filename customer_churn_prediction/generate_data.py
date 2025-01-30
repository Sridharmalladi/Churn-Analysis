import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_rows=1000):
    """Generate sample customer data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base data
    data = {
        'customer_id': [f'CUS_{str(i).zfill(5)}' for i in range(n_rows)],
        'account_created': [
            (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(n_rows)
        ],
        'last_purchase': [
            (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(n_rows)
        ],
        'total_purchase_amount': np.random.uniform(100, 5000, n_rows),
        'number_of_purchases': np.random.randint(1, 100, n_rows),
        'support_tickets': np.random.randint(0, 10, n_rows),
        'subscription_tier': np.random.choice(['basic', 'premium', 'enterprise'], n_rows),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal'], n_rows),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn based on realistic conditions
    df['churn'] = (
        ((df['support_tickets'] > 5) & (df['number_of_purchases'] < 10)) |
        (df['total_purchase_amount'] < 500) |
        ((df['support_tickets'] > 3) & (df['total_purchase_amount'] < 1000))
    ).astype(int)
    
    return df

if __name__ == "__main__":
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate sample data
    df = generate_sample_data(1000)
    
    # Save raw data
    df.to_csv('data/raw/customer_data.csv', index=False)
    print("Sample data generated and saved to 'data/raw/customer_data.csv'")