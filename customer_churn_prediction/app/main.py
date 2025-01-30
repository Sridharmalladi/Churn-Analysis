from flask import Flask, request, render_template, jsonify
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.predict import ChurnPredictor

app = Flask(__name__)
predictor = ChurnPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = {
            'account_age': float(data['account_age']),
            'purchase_frequency': float(data['purchase_frequency']),
            'avg_transaction_value': float(data['avg_transaction_value']),
            'support_tickets': int(data['support_tickets'])
        }
        
        result = predictor.predict(features)
        
        return jsonify({
            'status': 'success',
            'prediction': bool(result['prediction']),
            'churn_probability': float(result['churn_probability'])
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)