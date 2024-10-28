import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('models/final_xgboost_churn_model.pkl')

# Title of the app

st.title("Customer Churn Prediction App")

# Instructions
st.write("Enter customer details below to predict the likelihood of churn.")

# Collect user inputs as text fields
total_charges = st.text_input('Total Charges in dollars($)')
monthly_cost_per_year = st.text_input('Monthly Cost Per Year in dollars($)')
total_services = st.text_input('Total Services Subscribed (0-8)')
tenure_bucket = st.selectbox('Tenure Bucket', ('0-1 year', '1-2 years', '2-4 years', '4-6 years', '6+ years'))
contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Credit card (automatic)'))
internet_service = st.selectbox('Internet Service', ('Fiber optic', 'No', 'DSL'))

# Convert text inputs to numeric types, handling exceptions
try:
    total_charges = float(total_charges)
except ValueError:
    total_charges = 0.0  # Set to 0 or another default value if invalid

try:
    monthly_cost_per_year = float(monthly_cost_per_year)
except ValueError:
    monthly_cost_per_year = 0.0  # Set to 0 or another default value if invalid

try:
    total_services = int(total_services)
except ValueError:
    total_services = 0  # Set to 0 or another default value if invalid

# Calculate log-transformed TotalCharges for model input
total_charges_log = np.log1p(total_charges) if total_charges > 0 else 0

# One-hot encode categorical variables to match model training
contract_map = {'One year': [1, 0], 'Two year': [0, 1], 'Month-to-month': [0, 0]}
contract_encoded = contract_map[contract]

payment_method_map = {
    'Credit card (automatic)': [1, 0, 0],
    'Electronic check': [0, 1, 0],
    'Mailed check': [0, 0, 1]
}
payment_method_encoded = payment_method_map[payment_method]

internet_service_map = {'Fiber optic': [1, 0], 'No': [0, 1], 'DSL': [0, 0]}
internet_service_encoded = internet_service_map[internet_service]

tenure_bucket_map = {
    '0-1 year': [0, 0, 0],
    '1-2 years': [1, 0, 0],
    '2-4 years': [0, 1, 0],
    '4-6 years': [0, 0, 1],
    '6+ years': [0, 0, 0]
}
tenure_bucket_encoded = tenure_bucket_map[tenure_bucket]

# Create input DataFrame for prediction
input_data = pd.DataFrame([[
    total_charges_log, monthly_cost_per_year,
    *contract_encoded, *payment_method_encoded, *internet_service_encoded,
    total_services, *tenure_bucket_encoded
]], columns=[
    'TotalCharges_log', 'MonthlyCostPerYear', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'InternetService_Fiber optic', 'InternetService_No', 'TotalServices',
    'TenureBucket_1-2 years', 'TenureBucket_2-4 years', 'TenureBucket_4-6 years'
])

# Predict churn probability
if st.button('Predict Churn'):
    prediction = model.predict(input_data)
    churn_probability = model.predict_proba(input_data)[0][1]

    # Display results
    st.write("### Prediction: Churn" if prediction == 1 else "### Prediction: No Churn")
    st.write(f"**Churn Probability:** {churn_probability:.2f}")
