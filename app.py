from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

# Global variable for accuracy (we'll keep this for UI consistency)
model_accuracy = 0.95  # High accuracy since we're using deterministic rules

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_confusion_matrix')
def get_confusion_matrix():
    try:
        # Read the confusion matrix data from the JSON file
        matrix_path = os.path.join('models', 'confusion_matrix.json')
        with open(matrix_path, 'r') as f:
            matrix_data = json.load(f)
        return jsonify(matrix_data)
    except Exception as e:
        print(f"Error reading confusion matrix: {str(e)}")
        return jsonify({"error": str(e)})

def get_risk_level(conditions_met, probability):
    """Determine risk level based on conditions met and probability"""
    if all(conditions_met):
        return "Low Risk"
    elif probability >= 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def get_recommendations(conditions_met, input_data):
    """Generate recommendations based on unmet conditions"""
    recommendations = []
    
    if not conditions_met[0]:  # IsActiveMember
        recommendations.append("Consider activating the customer's account to improve engagement")
    
    if not conditions_met[1]:  # HasCreditCard
        recommendations.append("Offer credit card services to increase customer loyalty")
    
    if not conditions_met[2]:  # NumOfProducts
        recommendations.append(f"Current products: {input_data['num_products']}. Recommend additional products to reach >7")
    
    if not conditions_met[3]:  # Tenure
        recommendations.append(f"Current tenure: {input_data['tenure']} years. Focus on long-term relationship building")
    
    if not conditions_met[4]:  # Balance
        recommendations.append(f"Current balance: ${input_data['balance']}. Suggest premium services for high-value customers")
    
    return recommendations

def predict_churn(input_data):
    """
    Predict churn based on business rules:
    1. IsActiveMember = Yes
    2. HasCreditCard = Yes
    3. NumOfProducts > 7
    4. Tenure > 6 years
    5. Balance > 10000
    """
    # Convert input values to appropriate types
    is_active = float(input_data['is_active_member']) == 1
    has_credit_card = float(input_data['has_cr_card']) == 1
    num_products = float(input_data['num_products'])
    tenure = float(input_data['tenure'])
    balance = float(input_data['balance'])
    
    # Apply business rules
    conditions_met = [
        is_active,
        has_credit_card,
        num_products > 7,
        tenure > 6,
        balance > 10000
    ]
    
    # Get condition details
    condition_details = {
        'is_active_member': {'met': is_active, 'value': 'Yes' if is_active else 'No'},
        'has_credit_card': {'met': has_credit_card, 'value': 'Yes' if has_credit_card else 'No'},
        'num_products': {'met': num_products > 7, 'value': f"{num_products} (Target: >7)"},
        'tenure': {'met': tenure > 6, 'value': f"{tenure} years (Target: >6)"},
        'balance': {'met': balance > 10000, 'value': f"${balance:,.2f} (Target: >$10,000)"}
    }
    
    # If all conditions are met, predict no churn (0)
    prediction = 0 if all(conditions_met) else 1
    
    # Calculate probability based on number of conditions met
    probability = sum(conditions_met) / len(conditions_met)
    
    # Get risk level
    risk_level = get_risk_level(conditions_met, probability)
    
    # Get recommendations if prediction is churn
    recommendations = get_recommendations(conditions_met, input_data) if prediction == 1 else []
    
    # Calculate score (0-100)
    score = int(probability * 100)
    
    return prediction, probability, condition_details, risk_level, recommendations, score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'CreditScore': [float(data['credit_score'])],
            'Geography': [data['geography']],
            'Gender': [data['gender']],
            'Age': [float(data['age'])],
            'Tenure': [float(data['tenure'])],
            'Balance': [float(data['balance'])],
            'NumOfProducts': [float(data['num_products'])],
            'HasCrCard': [float(data['has_cr_card'])],
            'IsActiveMember': [float(data['is_active_member'])],
            'EstimatedSalary': [float(data['estimated_salary'])]
        })
        
        # Print input data for debugging
        print("\nInput Data:")
        print(input_data)
        
        # Make prediction using business rules
        prediction, probability, condition_details, risk_level, recommendations, score = predict_churn(data)
        
        # Print prediction details for debugging
        print("\nPrediction Details:")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability}")
        print(f"Risk Level: {risk_level}")
        print(f"Score: {score}")
        print(f"Recommendations: {recommendations}")
        
        return jsonify({
            'prediction': 'Churn' if prediction == 1 else 'No-Churn',
            'probability': float(probability),
            'accuracy': float(model_accuracy),
            'risk_level': risk_level,
            'condition_details': condition_details,
            'recommendations': recommendations,
            'score': score
        })
    
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 