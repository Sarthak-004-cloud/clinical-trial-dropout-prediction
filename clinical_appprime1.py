# Clinical Dropout Risk Prediction Dashboard
# Developed for AstraZeneca-style clinical trial datasets

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from streamlit.components.v1 import html
import os

# App configuration
st.set_page_config(page_title="Dropout Simulator", layout="wide")

# Utility to load models or scalers
def safe_load(path, fallback=None):
    return joblib.load(path) if os.path.exists(path) else fallback

# SHAP integration for Streamlit
def st_shap(plot, height=400):
    html(f"<head>{shap.getjs()}</head><body>{plot.html()}</body>", height=height)

# Load trained assets
model = safe_load("model.pkl")
scaler = safe_load("scaler.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", ["Upload & Predict", "Explainability", "Retention Strategy", "ROI Calculator"])

# App title
st.title("Clinical Trial Dropout Prediction & Retention Planning")
st.markdown("This dashboard analyzes patient dropout risk and supports intervention strategy decisions.")

# CSV Upload
uploaded_file = st.file_uploader("Upload your clinical trial data (CSV)", type=["csv"])
if not uploaded_file:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    if 'dwp_all' not in df.columns:
        st.error("Missing required column: 'dwp_all'")
        st.stop()

    df['dropout_flag'] = (df['dwp_all'] > 0).astype(int)
    patient_ids = df.get("nct_id", pd.Series([f"Trial_{i+1}" for i in range(len(df))]))

    input_df = df.drop(columns=[col for col in ['Unnamed: 0', 'nct_id', 'dwp_all', 'dropout_flag'] if col in df.columns])

    for col in input_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    if model and hasattr(model, 'feature_names_in_'):
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

    scaled_input = scaler.transform(input_df) if scaler else st.stop()
    probs = model.predict_proba(scaled_input)[:, 1]
    preds = model.predict(scaled_input)

    result_df = pd.DataFrame({
        "Trial_ID": patient_ids,
        "Dropout_Probability": np.round(probs, 3),
        "Prediction": preds
    })

    # Maintain threshold in session
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.5

    if section == "Upload & Predict":
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Prediction Results")
        st.dataframe(result_df, use_container_width=True)
        st.download_button("Download Predictions", result_df.to_csv(index=False), "predictions.csv")

        fig, ax = plt.subplots()
        sns.countplot(x='dropout_flag', data=df, ax=ax)
        ax.set_title("Dropout Distribution")
        st.pyplot(fig)

        if 'Duration.Trial' in df.columns:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='dropout_flag', y='Duration.Trial', data=df, ax=ax2)
            ax2.set_title("Duration vs Dropout")
            st.pyplot(fig2)

    elif section == "Explainability":
        try:
            explainer = shap.Explainer(model, scaled_input)
            shap_values = explainer(scaled_input)

            st.subheader("Feature Importance (SHAP Beeswarm)")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig3)

            st.subheader("Individual Prediction Explanation")
            st_shap(shap.plots.force(shap_values[0]), height=200)
        except Exception as e:
            st.warning(f"Unable to compute SHAP values: {e}")

    elif section == "Retention Strategy":
        st.subheader("Simulate Retention Interventions")
        threshold = st.slider("Dropout Risk Threshold", 0.0, 1.0, st.session_state["threshold"], 0.05, key="threshold_slider")
        st.session_state["threshold"] = threshold
        flagged = (probs >= threshold).sum()
        st.info(f"{flagged} patients exceed risk threshold of {threshold:.2f}")

        thresholds = np.linspace(0.1, 1.0, 10)
        flagged_counts = [(probs >= t).sum() for t in thresholds]

        fig4, ax4 = plt.subplots()
        ax4.plot(thresholds, flagged_counts, marker='o')
        ax4.set_title("Patients Flagged vs Risk Threshold")
        ax4.set_xlabel("Threshold")
        ax4.set_ylabel("Flagged Patients")
        st.pyplot(fig4)

    elif section == "ROI Calculator":
        st.subheader("Retention ROI Estimation")
        threshold = st.session_state.get("threshold", 0.5)
        flagged = (probs >= threshold).sum()

        cost_intervention = st.number_input("Cost per Intervention", value=1000, step=500)
        cost_dropout = st.number_input("Cost per Dropout", value=10000, step=1000)

        total_cost = flagged * cost_intervention
        avoided_loss = flagged * cost_dropout
        net_savings = avoided_loss - total_cost

        st.metric("Total Intervention Cost", f"₹{total_cost:,}")
        st.metric("Avoided Dropout Loss", f"₹{avoided_loss:,}")
        st.metric("Net Savings", f"₹{net_savings:,}")

        # Indian-style y-axis
        from matplotlib.ticker import FuncFormatter
        def format_indian(x, _):
            s = f"{int(x):,}".replace(",", "")
            if len(s) <= 3: return s
            last3, rest = s[-3:], s[:-3]
            formatted = ''
            while len(rest) > 2:
                formatted = "," + rest[-2:] + formatted
                rest = rest[:-2]
            return (rest + formatted + ',' + last3) if rest else formatted[1:] + ',' + last3

        fig5, ax5 = plt.subplots()
        bars = [total_cost, avoided_loss, net_savings]
        labels = ["Intervention Cost", "Avoided Loss", "Net Savings"]
        ax5.bar(labels, bars, color=['#007acc', '#28a745', '#ffc107'])
        ax5.set_title("Cost Comparison")
        ax5.set_ylabel("INR")
        ax5.yaxis.set_major_formatter(FuncFormatter(format_indian))
        st.pyplot(fig5)

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
