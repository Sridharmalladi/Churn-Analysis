import pytest
import pandas as pd
import numpy as np
from src.data.data_processing import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'customer_id': ['CUS_001', 'CUS_002', 'CUS_003'],
        'total_purchase_amount': [100.0, np.nan, 300.0],
        'number_of_purchases': [5, 3, 8],
        'support_tickets': [1, 2, np.nan],
        'subscription_tier': ['basic', 'premium', 'basic'],
        'churn': [0, 1, 0]
    })

def test_data_processor_initialization():
    """Test DataProcessor initialization"""
    processor = DataProcessor("dummy_path.csv")
    assert processor.filepath == "dummy_path.csv"
    assert processor.raw_data is None
    assert processor.processed_data is None

def test_clean_data(sample_data, tmp_path):
    """Test data cleaning functionality"""
    # Create temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Initialize processor and load data
    processor = DataProcessor(csv_path)
    processor.raw_data = sample_data
    cleaned_data = processor.clean_data()
    
    # Check if missing values are handled
    assert cleaned_data['total_purchase_amount'].isna().sum() == 0
    assert cleaned_data['support_tickets'].isna().sum() == 0
    
    # Check if categorical variables are encoded
    assert 'subscription_tier_basic' in cleaned_data.columns
    assert 'subscription_tier_premium' in cleaned_data.columns

def test_split_data(sample_data):
    """Test data splitting functionality"""
    processor = DataProcessor("dummy_path.csv")
    processor.raw_data = sample_data
    processor.processed_data = processor.clean_data()
    
    X_train, X_test, y_train, y_test = processor.split_data(test_size=0.2)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)