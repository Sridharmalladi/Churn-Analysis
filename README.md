# Customer Churn Prediction Project 🎯

## Overview
This project implements a machine learning system to predict customer churn in subscription-based businesses. Using historical customer data, the model identifies customers at risk of churning, allowing businesses to take proactive retention measures.

## Key Features 🌟
- Data preprocessing and cleaning pipeline
- Advanced feature engineering
- Multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Interactive web interface for real-time predictions
- Comprehensive test suite
- Detailed documentation and analysis

## Project Structure 📁
```
customer_churn_prediction/
├── app/                      # Web application files
│   ├── main.py              # Flask application
│   └── templates/           # HTML templates
├── data/                    # Data files
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── notebooks/               # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── src/                     # Source code
│   ├── data/               # Data processing scripts
│   ├── features/           # Feature engineering
│   └── models/             # Model training and prediction
├── tests/                  # Unit tests
└── requirements.txt        # Project dependencies
```

## Installation Guide 🚀

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer_churn_prediction.git
cd customer_churn_prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide 📖

### 1. Data Preparation
Generate sample data:
```bash
python generate_data.py
```

### 2. Training Pipeline
Run these scripts in sequence:
```bash
python src/data/data_processing.py
python src/features/feature_engineering.py
python src/models/train.py
```

### 3. Web Application
Start the Flask application:
```bash
python app/main.py
```
Access the web interface at `http://localhost:5000`

## Model Performance 📊
- Accuracy: 87%
- Precision: 85%
- Recall: 83%
- F1-Score: 84%

## Project Features 🛠️

### Data Processing
- Handles missing values
- Removes duplicates
- Encodes categorical variables
- Scales numerical features

### Feature Engineering
- Time-based features (account age, time since last purchase)
- Monetary features (average transaction value, monthly spend)
- Engagement metrics (purchase frequency, support ticket ratio)

### Model Training
- Multiple model comparison
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis

### Web Interface
- Real-time predictions
- User-friendly interface
- Detailed prediction explanations

## Testing 🧪
Run the test suite:
```bash
pytest tests/
```

## Contributing 🤝
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏
- scikit-learn documentation
- Flask documentation
- XGBoost documentation

