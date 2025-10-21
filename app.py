import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from datetime import datetime

# =========================================================
# 🌟 PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="💳 Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# 🎨 CUSTOM CSS STYLES
# =========================================================
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    color: #FFD700;
    text-shadow: 1px 1px 2px black;
}
.stApp {
    background: rgba(255,255,255,0);
}
div[data-testid="stSidebar"] {
    background: rgba(30, 30, 30, 0.9);
    color: white;
}
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}
.dataframe th, .dataframe td {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 TITLE AND INTRO
# =========================================================
st.title("💳 Fraud Detection System")
st.markdown("""
This AI-powered app can:  
- 📂 Predict fraud for uploaded transactions (CSV)  
- ✍️ Predict fraud for a manually entered transaction  
""")

# =========================================================
# 📦 LOAD MODEL AND FEATURES
# =========================================================
model = joblib.load("artifacts/fraud_xgb_pipeline.pkl")
feature_cols = joblib.load("artifacts/feature_cols.pkl")

# =========================================================
# 🔹 SECTION 1: CSV UPLOAD
# =========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("📂 Upload Transaction CSV File")

uploaded = st.file_uploader("Upload your transaction file (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["TX_DATETIME"])
    st.write("✅ Preview of uploaded data:")
    st.dataframe(df.head())

    def featurize(df):
        df = df.sort_values("TX_DATETIME").reset_index(drop=True)
        df["tx_hour"] = df["TX_DATETIME"].dt.hour
        df["tx_dayofweek"] = df["TX_DATETIME"].dt.dayofweek
        df["tx_is_weekend"] = df["tx_dayofweek"].isin([5,6]).astype(int)
        df["tx_amount_log"] = np.log1p(df["TX_AMOUNT"])
        df["cust_tx_count_prior"] = df.groupby("CUSTOMER_ID").cumcount()
        df["cust_tx_amount_mean_prior"] = (
            df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .apply(lambda x: x.shift().expanding().mean().fillna(0))
        )
        df["term_tx_count_prior"] = df.groupby("TERMINAL_ID").cumcount()
        df["term_tx_amount_mean_prior"] = (
            df.groupby("TERMINAL_ID")["TX_AMOUNT"]
            .apply(lambda x: x.shift().expanding().mean().fillna(0))
        )
        df["tx_amount_over_cust_avg"] = df["TX_AMOUNT"] / df["cust_tx_amount_mean_prior"].replace(0, np.nan)
        df["tx_amount_over_term_avg"] = df["TX_AMOUNT"] / df["term_tx_amount_mean_prior"].replace(0, np.nan)
        for col in ["tx_amount_over_cust_avg", "tx_amount_over_term_avg"]:
            df[col].replace([np.inf, -np.inf], 0, inplace=True)
            df[col].fillna(0, inplace=True)
        df["flag_amount_gt_220"] = (df["TX_AMOUNT"] > 220).astype(int)
        X = df[feature_cols].fillna(0)
        return df, X

    df_proc, X = featurize(df)
    proba = model.predict_proba(X)[:,1]
    pred = model.predict(X)
    df_proc["fraud_proba"] = proba
    df_proc["fraud_pred"] = pred

    st.success("✅ Predictions generated successfully!")
    st.dataframe(df_proc[["TRANSACTION_ID", "TX_DATETIME", "TX_AMOUNT", "fraud_proba", "fraud_pred"]].head(15))
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 🔸 SECTION 2: MANUAL TRANSACTION INPUT
# =========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.header("✍️ Manual Transaction Input")

col1, col2, col3 = st.columns(3)

with col1:
    tx_amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, step=1.0)
    tx_datetime = st.date_input("🗓 Transaction Date", datetime(2018, 4, 1))
    tx_hour = st.number_input("⏰ Hour (0-23)", min_value=0, max_value=23, value=12)
with col2:
    customer_id = st.number_input("🧍 Customer ID", min_value=0)
    terminal_id = st.number_input("🏧 Terminal ID", min_value=0)
    cust_avg = st.number_input("📊 Customer Avg. Spending", min_value=0.0, value=100.0)
with col3:
    term_avg = st.number_input("🏦 Terminal Avg. Spending", min_value=0.0, value=150.0)
    cust_tx_count = st.number_input("🔢 Customer Transaction Count", min_value=0, value=10)
    term_tx_count = st.number_input("🔢 Terminal Transaction Count", min_value=0, value=20)

# Derived features
tx_dayofweek = pd.Timestamp(tx_datetime).dayofweek
tx_is_weekend = 1 if tx_dayofweek in [5,6] else 0
tx_amount_log = np.log1p(tx_amount)
tx_amount_over_cust_avg = tx_amount / cust_avg if cust_avg > 0 else 0
tx_amount_over_term_avg = tx_amount / term_avg if term_avg > 0 else 0
flag_amount_gt_220 = 1 if tx_amount > 220 else 0

input_df = pd.DataFrame([{
    "TX_AMOUNT": tx_amount,
    "tx_amount_log": tx_amount_log,
    "tx_hour": tx_hour,
    "tx_dayofweek": tx_dayofweek,
    "tx_is_weekend": tx_is_weekend,
    "cust_tx_count_prior": cust_tx_count,
    "cust_tx_amount_mean_prior": cust_avg,
    "term_tx_count_prior": term_tx_count,
    "term_tx_amount_mean_prior": term_avg,
    "tx_amount_over_cust_avg": tx_amount_over_cust_avg,
    "tx_amount_over_term_avg": tx_amount_over_term_avg,
    "flag_amount_gt_220": flag_amount_gt_220
}])[feature_cols]

if st.button("🚀 Predict Fraud"):
    with st.spinner("Analyzing transaction with AI..."):
        proba = model.predict_proba(input_df)[:,1][0]
        pred = model.predict(input_df)[0]
    st.subheader("🧾 Prediction Result:")
    st.metric(label="Fraud Probability", value=f"{proba*100:.2f}%")
    st.metric(label="Status", value="🚨 Fraud Detected" if pred==1 else "✅ Legitimate Transaction")

    try:
        explainer = shap.TreeExplainer(model.named_steps["clf"])
        preproc = model.named_steps["preproc"]
        X_pre = preproc.transform(input_df)
        shap_values = explainer.shap_values(pd.DataFrame(X_pre, columns=feature_cols))
        shap_df = pd.DataFrame({"Feature": feature_cols, "Impact": shap_values[0]})
        shap_df["AbsImpact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values("AbsImpact", ascending=False)
        st.write("📊 Top Influential Features (SHAP):")
        st.dataframe(shap_df.head(10))
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation skipped: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 🧾 FOOTER
# =========================================================
st.markdown("""
---
💡 *Developed by Rounak | Powered by XGBoost + Streamlit*  
""")
