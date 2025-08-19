import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Paths (relative to your app.py location)
DATA_PATH = "dataset/credit_risk_dataset.csv"
MODEL_PATH = "credit_risk_model.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(df):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Encode categorical columns and save encoders
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model and encoders together
    joblib.dump((model, encoders, target_encoder), MODEL_PATH)
    return model, encoders, target_encoder

@st.cache_resource
def load_model():
    model, encoders, target_encoder = joblib.load(MODEL_PATH)
    return model, encoders, target_encoder

# Load data
df = load_data()

# Load or train model
if not os.path.exists(MODEL_PATH):
    model, encoders, target_encoder = train_model(df)
else:
    model, encoders, target_encoder = load_model()

st.title("Credit Risk Prediction for Loan Approval")

# Prepare input widgets
X = df.drop('loan_status', axis=1)
inputs = {}

for col in X.columns:
    if X[col].dtype == 'object' or X[col].nunique() <= 10:
        options = X[col].unique().tolist()
        inputs[col] = st.selectbox(f"Select {col}", options)
    else:
        min_val = int(X[col].min())
        max_val = int(X[col].max())
        inputs[col] = st.number_input(f"Input {col}", min_value=min_val, max_value=max_val, value=min_val)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# Apply the same label encoding as training
for col, le in encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Prediction button
if st.button("Predict"):
    prediction_encoded = model.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"Loan Status: {prediction_label}")
