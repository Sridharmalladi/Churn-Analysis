import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'customer_id': ['CUS_001', 'CUS_002'],
        'account_created': ['2023-01-01', '2023-01-02'],
        'last_purchase': ['2023-06-01', '2023-06-02'],
        'total_purchase_amount': [1000.0, 2000.0],
        'number_of_purchases': [10, 20],
        'support_tickets': [2, 4]
    })

def test_feature_engineer_initialization(sample_data):
    """Test FeatureEngineer initialization"""
    engineer = FeatureEngineer(sample_data)
    assert engineer.df is not None
    assert len(engineer.df) == len(sample_data)
    assert all(engineer.df.columns == sample_data.columns)

def test_create_time_features(sample_data):
    """Test time-based feature creation"""
    engineer = FeatureEngineer(sample_data)
    result = engineer.create_time_features()
    
    # Check if new features are created
    assert 'account_age_days' in result.columns
    assert 'days_since_last_purchase' in result.columns
    
    # Check if values are calculated correctly
    assert all(result['account_age_days'] > 0)
    assert all(result['days_since_last_purchase'] > 0)

def test_create_monetary_features(sample_data):
    """Test monetary feature creation"""
    engineer = FeatureEngineer(sample_data)
    engineer.create_time_features()  # Required for some monetary features
    result = engineer.create_monetary_features()
    
    # Check if new features are created
    assert 'avg_transaction_value' in result.columns
    assert 'monthly_spend' in result.columns
    
    # Check calculations
    expected_avg = sample_data['total_purchase_amount'] / sample_data['number_of_purchases']
    pd.testing.assert_series_equal(
        result['avg_transaction_value'],
        expected_avg,
        check_names=False
    )

def test_create_engagement_features(sample_data):
    """Test engagement feature creation"""
    engineer = FeatureEngineer(sample_data)
    engineer.create_time_features()  # Required for engagement features
    result = engineer.create_engagement_features()
    
    # Check if new features are created
    assert 'purchase_frequency' in result.columns
    assert 'support_ticket_ratio' in result.columns
    
    # Check if values are in expected range
    assert all(result['purchase_frequency'] >= 0)
    assert all(result['support_ticket_ratio'] >= 0)

def test_engineer_features_pipeline(sample_data):
    """Test complete feature engineering pipeline"""
    engineer = FeatureEngineer(sample_data)
    result = engineer.engineer_features()
    
    # Check if all expected features are present
    expected_features = [
        'account_age_days', 'days_since_last_purchase',
        'avg_transaction_value', 'monthly_spend',
        'purchase_frequency', 'support_ticket_ratio'
    ]
    
    for feature in expected_features:
        assert feature in result.columns