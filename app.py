"""
app.py
------
Streamlit web app to demo the churn prediction model interactively.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

model = joblib.load("outputs/churn_model.pkl")

st.title("📉 Customer Churn Prediction")
st.write("Enter customer details to predict the probability that they will churn.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

if st.button("Predict Churn"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])

    proba = model.predict_proba(row)[0, 1]
    pred = "Yes" if proba >= 0.5 else "No"

    st.subheader(f"Prediction: {'⚠️ Likely to Churn' if pred == 'Yes' else '✅ Likely to Stay'}")
    st.metric("Churn Probability", f"{proba*100:.1f}%")
    st.progress(min(proba, 1.0))
