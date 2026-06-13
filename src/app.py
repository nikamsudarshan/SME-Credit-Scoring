# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(page_title="SME XAI Underwriter", layout="wide")
st.title("🏦 SME Alternative Credit Scoring Engine")
st.markdown("Evaluating MSME loan eligibility using Hybrid Data & Explainable AI (SHAP) for RBI Compliance.")

# --- Background Model Training ---
@st.cache_resource
def initialize_engine():
    # 1. Fetch & Clean SBA Data
    url = "https://raw.githubusercontent.com/stelladeecoder/sba_dataset/main/SBAcase.11.13.17.csv"
    urllib.request.urlretrieve(url, "SBAcase.csv")
    df = pd.read_csv("SBAcase.csv")

    df['Default'] = np.where(df['MIS_Status'] == 'CHGOFF', 1, 0)

    # Fixed the regex syntax warning using 'r' for raw string
    for col in ['DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

    core_features = ['DisbursementGross', 'Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'Default']
    df = df[core_features].dropna().copy()

    # 2. Inject Alternative Features
    np.random.seed(42)
    df['GST_Consistency_Pct'] = np.where(df['Default'] == 1, np.random.normal(40, 20, len(df)), np.random.normal(85, 10, len(df)))
    df['GST_Consistency_Pct'] = np.clip(df['GST_Consistency_Pct'], 0, 100)

    df['Social_Sentiment_Score'] = np.where(df['Default'] == 1, np.random.normal(-2, 3, len(df)), np.random.normal(5, 2, len(df)))
    df['Social_Sentiment_Score'] = np.clip(df['Social_Sentiment_Score'], -10, 10)

    df['Local_Market_Growth_Pct'] = np.where(df['Default'] == 1, np.random.normal(2, 4, len(df)), np.random.normal(6, 3, len(df)))

    # 3. Train the XGBoost Model
    X = df.drop('Default', axis=1)
    y = df['Default']

    model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # Initialize the SHAP Explainer
    explainer = shap.Explainer(model)

    return model, explainer, X.columns

with st.spinner("Training ML Engine & Loading Data..."):
    model, explainer, feature_names = initialize_engine()

# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Applicant Data")
    st.markdown("Adjust parameters to simulate an SME application.")

    # Traditional Financials
    st.subheader("Financial Request")
    loan_amount = st.number_input("Requested Loan Amount (₹/USD)", min_value=1000, max_value=5000000, value=50000, step=5000)
    term = st.slider("Loan Term (Months)", min_value=12, max_value=360, value=84)
    employees = st.slider("Current Employees", min_value=1, max_value=100, value=5)
    create_job = st.slider("Jobs Created", min_value=0, max_value=50, value=2)
    retained_job = st.slider("Jobs Retained", min_value=0, max_value=50, value=5)

    # Alternative Metrics
    st.subheader("Alternative Metrics")
    gst_pct = st.slider("GST Filing Consistency (%)", min_value=0, max_value=100, value=90)
    social_score = st.slider("Social Media Sentiment (-10 to +10)", min_value=-10.0, max_value=10.0, value=4.5)
    market_growth = st.slider("Local Market Growth Rate (%)", min_value=-5.0, max_value=15.0, value=5.5)

with col2:
    st.header("📊 AI Underwriting Decision")

    # Create the 1-row dataframe for prediction
    input_data = pd.DataFrame([[
        loan_amount, term, employees, create_job, retained_job,
        gst_pct, social_score, market_growth
    ]], columns=feature_names)

    # Get Probability of Default (Class 1)
    default_prob = model.predict_proba(input_data)[0][1]

    # Logic for Approval
    if default_prob < 0.20:
        st.success(f"✅ **STATUS: APPROVED** | Default Risk: {default_prob:.1%}")
        st.markdown("This business represents a **Low Risk**. Their alternative metrics heavily offset any lack of traditional credit.")
    elif default_prob < 0.50:
        st.warning(f"⚠️ **STATUS: MANUAL REVIEW REQUIRED** | Default Risk: {default_prob:.1%}")
        st.markdown("This business represents a **Moderate Risk**. Request additional collateral or a co-signer.")
    else:
        st.error(f"🚫 **STATUS: REJECTED** | Default Risk: {default_prob:.1%}")
        st.markdown("This business represents a **High Risk**. The combination of financials and alternative metrics indicates instability.")

    st.markdown("---")
    st.subheader("🔍 Explainable AI (XAI) Reasoning")
    st.markdown("The chart below maps exactly how the AI reached its decision. Blue bars push the risk down (Safe), Red bars push the risk up (Default).")

    # Generate SHAP Plot for the specific input
    shap_values = explainer(input_data)

    # Plotting wrapper for Streamlit
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
