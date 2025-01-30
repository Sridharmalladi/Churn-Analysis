import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def test_data_path(tmp_path_factory):
    """Create a temporary directory for test data"""
    return tmp_path_factory.mktemp("data")

@pytest.fixture(scope="session")
def sample_customer_data():
    """Generate sample customer data for testing"""
    return pd.DataFrame({
        'customer_id': [f'CUS_{i:03d}' for i in range(100)],
        'account_created': [(datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d') 
                           for _ in range(100)],
        'last_purchase': [(datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d') 
                         for _ in range(100)],
        'total_purchase_amount': np.random.uniform(100, 1000, 100),
        'number_of_purchases': np.random.randint(1, 50, 100),
        'support_tickets': np.random.randint(0, 10, 100),
        'subscription_tier': np.random.choice(['basic', 'premium', 'enterprise'], 100),
        'churn': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    })

@pytest.fixture
def mock_model_trainer(mocker):
    """Mock model trainer for testing"""
    return mocker.patch('src.models.train.ModelTrainer')

@pytest.fixture
def mock_predictor(mocker):
    """Mock predictor for testing"""
    return mocker.patch('src.models.predict.ChurnPredictor')