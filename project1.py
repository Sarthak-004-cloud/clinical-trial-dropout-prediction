#!/usr/bin/env python
# coding: utf-8

# # Clinical Trial Dropout Prediction & Retention Strategy Simulator
# 
# **Objective**: Predict which patients are likely to drop out of clinical trials using machine learning, explain the key influencing factors using SHAP, and simulate personalized retention strategies.
# 
# **Why this matters**: Reducing dropout rates improves trial success, cost efficiency, and patient outcomes — a major challenge in the healthcare industry.
# 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
df = pd.read_csv("clinical_trial_data.csv") 

# Preprocessing
X = df.drop("dropout", axis=1) 
y = df["dropout"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Optional: Save for later
joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import shap
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
df = pd.read_csv("clinical_trial_data.csv")

# Show first few rows
df.head()




# Shape of the dataset
print("Rows, Columns:", df.shape)

# Data types and null values
print("\nInfo:")
print(df.info())

# Basic summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Count of target variable (dropout)
print("\nDropout Distribution:")
print(df['dropout'].value_counts())


# CORRELATION HEATMAP



plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#  Dropout Count Plot



sns.countplot(x='dropout', data=df)
plt.title("Dropout Distribution")
plt.xlabel("Dropout (1 = Dropped out)")
plt.ylabel("Count")
plt.show()


# Boxplot: Missed Visits vs Dropout




sns.boxplot(x='dropout', y='visits_missed', data=df)
plt.title("Missed Visits vs Dropout")
plt.show()


# Distribution by Age



plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='age', hue='dropout', bins=20, kde=True)
plt.title("Age Distribution by Dropout")
plt.show()


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')




sns.countplot(x='dropout', data=df)




sns.boxplot(x='dropout', y='visits_missed', data=df)




sns.histplot(data=df, x='age', hue='dropout', bins=20, kde=True)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Check for missing values
print("Missing values:\n", df.isnull().sum())

# OPTIONAL: If missing values exist, you can fill or drop them
# For now, let’s drop rows with missing values for simplicity
df.dropna(inplace=True)

# 2. Encode categorical variables (e.g., gender, side_effects if categorical)
label_cols = df.select_dtypes(include='object').columns

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# 3. Feature-Target split
X = df.drop('dropout', axis=1)
y = df['dropout']

# 4. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Preprocessing complete!")




# Importing required modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize models
lr = LogisticRegression()
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train models
lr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

# Predict
lr_pred = lr.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)
xgb_pred = xgb.predict(X_test_scaled)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n {name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate all
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("XGBoost", y_test, xgb_pred)




import shap

# Initialize SHAP explainer for logistic regression model
explainer = shap.Explainer(log_model, X_test_scaled)

# Calculate SHAP values
shap_values = explainer(X_test_scaled)



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the data again (if scaler isn't already defined)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)



import shap

# Initialize SHAP explainer
explainer = shap.Explainer(log_model, X_test_scaled)

# Calculate SHAP values
shap_values = explainer(X_test_scaled)




shap.plots.beeswarm(shap_values, max_display=10)




shap.initjs()  # Enable JS for visualization
shap.plots.force(shap_values[0])




shap.plots.bar(shap_values)




def simulate_retention_intervention(shap_values, threshold=0.3):
    """
    Simulate improved outcomes by applying retention support
    to patients whose dropout SHAP score exceeds a threshold.
    """
    retained = 0
    for i in range(len(shap_values)):
        dropout_risk = shap_values[i].values[1]  # 1 = 'dropout'
        if dropout_risk > threshold:
            retained += 1  # Assume retention works
    return retained




import numpy as np
import matplotlib.pyplot as plt

def simulate_retention(shap_values, threshold_range=np.linspace(0.1, 1.0, 10)):
    """
    Simulates the number of patients saved from dropout by applying
    interventions at different SHAP-based risk thresholds.

    Args:
        shap_values: SHAP values object from explainer.shap_values(X_test)
        threshold_range: Array of SHAP thresholds to simulate

    Returns:
        thresholds, saved_counts: Arrays of thresholds vs patients saved
    """
    dropout_shap_scores = shap_values[1].sum(axis=1)  # Class 1 = dropout
    saved_counts = []

    for threshold in threshold_range:
        saved = np.sum(dropout_shap_scores >= threshold)
        saved_counts.append(saved)

    return threshold_range, saved_counts




def plot_retention_simulation(thresholds, saved_counts):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, saved_counts, marker='o', color='crimson')
    plt.title("Simulated Patients Retained vs SHAP Risk Threshold")
    plt.xlabel("SHAP Risk Threshold (Dropout Likelihood)")
    plt.ylabel("Patients Retained by Intervention")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



explainer = shap.TreeExplainer(log_model)
shap_values = explainer.shap_values(X_test)




explainer = shap.Explainer(log_model, X_train)
shap_values = explainer(X_test)



explainer = shap.Explainer(log_model, X_train)
shap_values = explainer(X_test)




thresholds, saved_counts = simulate_retention(shap_values)
plot_retention_simulation(thresholds, saved_counts)




def simulate_retention(shap_values, threshold_range=np.linspace(0.1, 1.0, 10)):
    """
    Simulate number of patients saved (i.e., predicted high dropout risk)
    by applying different SHAP score thresholds.
    """
    # shap_values.values is a (n_samples,) array of total SHAP impact scores
    dropout_shap_scores = np.abs(shap_values.values).sum(axis=1)

    saved_counts = []
    for threshold in threshold_range:
        count = np.sum(dropout_shap_scores >= threshold)
        saved_counts.append(count)

    return threshold_range, saved_counts




thresholds, saved_counts = simulate_retention(shap_values)
plot_retention_simulation(thresholds, saved_counts)



def simulate_retention(shap_values, threshold_range=np.linspace(0.1, 1.0, 10)):
    """
    Simulates retention intervention based on SHAP predicted dropout scores.
    """
    # SHAP output prediction = base + shap value sum
    dropout_risk_scores = shap_values.base_values + shap_values.values.sum(axis=1)

    saved_counts = []
    for threshold in threshold_range:
        count = np.sum(dropout_risk_scores >= threshold)
        saved_counts.append(count)

    return threshold_range, saved_counts


thresholds, saved_counts = simulate_retention(shap_values)
plot_retention_simulation(thresholds, saved_counts)





def simulate_retention_v2(y_probs, threshold_range=np.linspace(0.1, 1.0, 10)):
    """
    Simulates number of patients flagged as high risk at various probability thresholds.
    Uses model's predicted dropout probabilities instead of SHAP scores.
    """
    saved_counts = []
    for threshold in threshold_range:
        count = np.sum(y_probs >= threshold)
        saved_counts.append(count)
    return threshold_range, saved_counts





# Get dropout probabilities (class 1 = dropout)
dropout_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# Simulate retention
thresholds, saved_counts = simulate_retention_v2(dropout_probs)

# Plot
plot_retention_simulation(thresholds, saved_counts)




shap.summary_plot(shap_values.values, X_test, plot_type="bar")




shap.summary_plot(shap_values.values, X_test)




shap.plots.waterfall(shap_values[0])  # Change index to view others




for i in range(3):
    shap.plots.waterfall(shap_values[i])





import joblib

joblib.dump(log_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")




joblib.dump(explainer, "shap_explainer.pkl")

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height or 500, scrolling=True)


st.title(" Clinical Trial Dropout Prediction")
st.markdown("Upload patient data (CSV) to predict dropout risk and get SHAP explanations.")

# Upload CSV
uploaded_file = st.file_uploader(" Upload Patient CSV File", type=["csv"])

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)

    # Show original uploaded data (including Patient_ID)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(uploaded_df)

    # Save Patient_ID if present
    if 'Patient_ID' in uploaded_df.columns:
        patient_ids = uploaded_df['Patient_ID']
    else:
        patient_ids = pd.Series([f"Patient_{i}" for i in range(len(uploaded_df))])

    # Keep only numeric data for model input
    X_input = uploaded_df.select_dtypes(include=[np.number])

    # Scale input
    X_input_scaled = scaler.transform(X_input)

    # Predict dropout
    predictions = model.predict(X_input_scaled)
    proba = model.predict_proba(X_input_scaled)[:, 1]  # probability of dropout

    # SHAP explanation
    explainer = shap.Explainer(model, X_input_scaled)
    shap_values = explainer(X_input_scaled)

    # Show results
    output_df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Dropout_Predicted (0=No, 1=Yes)": predictions,
        "Dropout_Risk_Probability": np.round(proba, 2)
    })

    st.subheader(" Prediction Results")
    st.dataframe(output_df)

    # Optional SHAP force plots for first 5 patients
    st.subheader(" SHAP Explanation (Top 5 Patients)")
    for i in range(min(5, len(uploaded_df))):
        st.markdown(f"#### Patient: {patient_ids.iloc[i]}")
        st_shap(shap.plots.force(shap_values[i], matplotlib=True), height=100)





import shap
explainer = shap.LinearExplainer(log_model, X_train_scaled)
joblib.dump(explainer, "shap_explainer.pkl")




import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Clinical Trial Dropout Predictor", layout="centered")

st.title(" Clinical Trial Dropout Prediction")
st.markdown("Upload patient data to **predict dropout risk** and explain results using **SHAP**.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with patient features", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("**Uploaded Data Preview:**", data.head())

    # Scale the input
    scaled_data = scaler.transform(data)

    # Predict
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]

    data["Dropout Risk (Probability)"] = probabilities
    data["Prediction"] = ["Likely to Stay" if p < 0.5 else "Likely to Dropout" for p in probabilities]

    st.write(" **Predictions:**", data[["Dropout Risk (Probability)", "Prediction"]])

    # SHAP Explainability
    st.subheader(" SHAP Explainability (First Patient)")

    try:
        explainer = shap.Explainer(model, scaled_data)
        shap_values = explainer(scaled_data)

        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP could not explain this model: {e}")

