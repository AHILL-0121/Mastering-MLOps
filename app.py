"""
app.py - Streamlit deployment for UK Retail Purchase Prediction
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Retail Purchase Predictor",
    page_icon="🛒",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "xgb_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Please run `train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 UK Retail Purchase Predictor")
st.markdown(
    """
    Predict whether a UK customer will make a purchase **next month**
    based on their current transaction behaviour.
    """
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Customer Transaction Details")

col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input(
        "Quantity (units purchased)",
        min_value=1, max_value=10_000, value=10, step=1
    )
    unit_price = st.number_input(
        "Unit Price (£)",
        min_value=0.01, max_value=10_000.0, value=2.50, step=0.01, format="%.2f"
    )

with col2:
    invoice_hour = st.slider(
        "Invoice Hour (0–23)", min_value=0, max_value=23, value=12
    )
    invoice_day = st.slider(
        "Invoice Day (1–31)", min_value=1, max_value=31, value=15
    )

total_price = quantity * unit_price
st.markdown(f"**🧾 Computed Total Price:** £ {total_price:,.2f}")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Purchase Likelihood", use_container_width=True):
    features = pd.DataFrame([{
        "Quantity"   : quantity,
        "UnitPrice"  : unit_price,
        "TotalPrice" : total_price,
        "InvoiceHour": invoice_hour,
        "InvoiceDay" : invoice_day,
    }])

    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    prob_buy   = round(probability[1] * 100, 1)
    prob_nobuy = round(probability[0] * 100, 1)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Purchase Likely** — {prob_buy}% confidence")
    else:
        st.warning(f"❌ **Not Likely to Purchase** — {prob_nobuy}% confidence of no purchase")

    # Probability bar
    st.markdown("**Probability Breakdown**")
    prob_df = pd.DataFrame({
        "Outcome"    : ["Will Purchase", "Will NOT Purchase"],
        "Probability": [prob_buy / 100, prob_nobuy / 100],
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    # Feature echo
    with st.expander("📎 Input features used"):
        st.dataframe(features, use_container_width=True)

# ── Batch prediction ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📂 Batch Prediction (CSV Upload)")
st.markdown(
    "Upload a CSV with columns: `Quantity`, `UnitPrice`, `TotalPrice`, "
    "`InvoiceHour`, `InvoiceDay`"
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    batch_df = pd.read_csv(uploaded)
    required = ["Quantity", "UnitPrice", "TotalPrice", "InvoiceHour", "InvoiceDay"]
    missing  = [c for c in required if c not in batch_df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        preds = model.predict(batch_df[required])
        probs = model.predict_proba(batch_df[required])[:, 1]
        batch_df["Prediction"]   = preds
        batch_df["Probability"]  = probs.round(4)
        batch_df["Outcome"]      = batch_df["Prediction"].map(
            {1: "Purchase Likely", 0: "Not Likely"}
        )
        st.dataframe(batch_df, use_container_width=True)
        csv = batch_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("MLOps Level 0–2 Demo | XGBoost + Streamlit | UK Online Retail II Dataset")
