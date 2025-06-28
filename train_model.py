# --- Model Retraining for Real Dataset Compatibility ---
# This script should be run once manually to align the model with AstraZeneca dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load real dataset
df = pd.read_csv("Indexed_Analysis_Ready.csv")

# Drop irrelevant columns
df = df.drop(columns=[col for col in ['Unnamed: 0', 'nct_id'] if col in df.columns])

# Create binary target
df['dropout_flag'] = (df['dwp_all'] > 0).astype(int)

# Define features and target
X = df.drop(columns=['dropout_flag', 'dwp_all'])
y = df['dropout_flag']

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save trained model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully. Now you can run the Streamlit app.")
