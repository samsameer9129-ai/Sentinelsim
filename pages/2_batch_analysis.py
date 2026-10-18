
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SIMGuard | Batch Analysis",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SIMGuard — Batch Fraud Analysis")
st.write(
    "Upload a CSV containing multiple transactions to assess "
    "their simulated SIM-swapping fraud risk."
)

# Locate the trained model from the project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "sim_swap_model.joblib"

FEATURES = [
    "hours_since_sim_change",
    "sim_change_detected",
    "new_device",
    "failed_logins",
    "transaction_amount",
    "average_transaction_amount",
    "transaction_count_24h",
    "unusual_location",
    "recent_credential_change",
]

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.info("Run train_model.py first to generate the model.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]

st.info(
    "This model was trained on synthetic data. Its risk scores "
    "are experimental and are not validated real-world probabilities."
)

st.subheader("1. Upload transaction data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    st.write("**Uploaded data preview**")
    st.dataframe(df.head(10), use_container_width=True)

    missing = [column for column in FEATURES if column not in df.columns]

    if missing:
        st.error("Your CSV is missing these required columns:")
        st.write(missing)
        st.write("Use the same feature names as the training dataset.")
        st.stop()

    if df.empty:
        st.warning("The uploaded CSV contains no transactions.")
        st.stop()

    if len(df) > 100_000:
        st.error("For this prototype, upload at most 100,000 rows at a time.")
        st.stop()

    # Keep only the model features and convert them to numeric values.
    X = df[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Reject invalid non-missing values instead of silently treating
    # malformed input as ordinary missing data.
    for column in FEATURES:
        original = df[column]
        converted = X[column]

        invalid = original.notna() & converted.isna()
        if invalid.any():
            st.error(
                f"Column '{column}' contains non-numeric values. "
                "Please correct them and upload the CSV again."
            )
            st.stop()

    if X.isna().all(axis=1).any():
        st.error(
            "At least one row has no usable feature values. "
            "Please correct or remove completely empty transaction rows."
        )
        st.stop()

    if st.button("Analyze transactions", type="primary"):
        try:
            probabilities = model.predict_proba(X)

            # Find the probability column for class 1 (fraud).
            classes = list(model.classes_)
            fraud_index = classes.index(1)
            scores = probabilities[:, fraud_index]

            results = df.copy()
            results["fraud_risk_score_pct"] = (scores * 100).round(2)
            results["risk_level"] = pd.cut(
                scores,
                bins=[-0.000001, 0.30, 0.70, 1.000001],
                labels=["Low", "Medium", "High"],
            ).astype(str)

            results = results.sort_values(
                "fraud_risk_score_pct",
                ascending=False,
            )

            st.session_state["batch_results"] = results

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

if "batch_results" in st.session_state:
    results = st.session_state["batch_results"]

    st.divider()
    st.subheader("2. Analysis results")

    total = len(results)
    high = int((results["risk_level"] == "High").sum())
    medium = int((results["risk_level"] == "Medium").sum())
    low = int((results["risk_level"] == "Low").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions analyzed", f"{total:,}")
    col2.metric("High risk", f"{high:,}")
    col3.metric("Medium risk", f"{medium:,}")
    col4.metric("Low risk", f"{low:,}")

    st.caption(
        "Risk bands are dashboard display categories, not validated "
        "operational thresholds."
    )

    st.subheader("Risk distribution")
    st.bar_chart(results["risk_level"].value_counts())

    st.subheader("Transactions ranked by risk")
    st.dataframe(results, use_container_width=True, height=400)

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download analyzed transactions",
        data=csv_data,
        file_name="simguard_batch_results.csv",
        mime="text/csv",
    )
