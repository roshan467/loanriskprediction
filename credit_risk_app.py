import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# Paths
DATA_PATH = r"C:\Users\Roshan\OneDrive\Desktop\credit_risk_project\dataset\credit_risk_dataset.csv"
MODEL_PATH = r"C:\Users\Roshan\OneDrive\Desktop\credit_risk_project\credit_risk_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Train model if not exists
if not os.path.exists(MODEL_PATH):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    categorical_cols = X.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    y = LabelEncoder().fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

st.title("Credit Risk Prediction for Loan Approval")

# Prepare input widgets
X = df.drop('loan_status', axis=1)
inputs = {}
for col in X.columns:
    if df[col].dtype == 'object' or df[col].nunique() <= 10:
        options = df[col].unique().tolist()
        inputs[col] = st.selectbox(col, options)
    else:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        inputs[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=min_val)

input_df = pd.DataFrame([inputs])

categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])
    input_df[col] = le.transform(input_df[col])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.success(f"Loan Status: {result}")
