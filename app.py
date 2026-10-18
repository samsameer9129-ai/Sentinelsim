
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. App configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sim_swap_model.joblib"

st.set_page_config(
    page_title="SIMGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# Dark, professional dashboard styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }
    [data-testid="stMetric"] {
        background: #151f31;
        border: 1px solid #26364c;
        padding: 16px;
        border-radius: 12px;
    }
    div.stButton > button {
        background-color: #0d9488;
        color: white;
        border-radius: 8px;
        border: none;
        min-height: 44px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #0f766e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()
    model = artifact["model"]
    features = artifact["features"]
    threshold = artifact.get("threshold", 0.50)
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# --------------------------------------------------
# 3. Dashboard header
# --------------------------------------------------

st.title("🛡️ SIMGuard")
st.subheader("SIM-Swapping Fraud Detection System")

st.caption(
    "Machine-learning prototype for identifying transactions "
    "that exhibit suspicious SIM-change and account behavior."
)

st.info(
    "Demonstration mode: the current model was trained on synthetic "
    "data. Its risk scores are not validated estimates of real-world "
    "fraud probability."
)


# --------------------------------------------------
# 4. Transaction input
# --------------------------------------------------

st.header("Analyze a transaction")

with st.form("transaction_form"):
    left, right = st.columns(2)

    with left:
        hours_since_sim_change = st.number_input(
            "Hours since SIM change",
            min_value=0.0,
            max_value=720.0,
            value=2.0,
            step=1.0,
            help="Elapsed hours since the most recent known SIM change.",
        )

        sim_change_detected = st.selectbox(
            "SIM change detected?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        new_device = st.selectbox(
            "Is this a new device?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        failed_logins = st.number_input(
            "Recent failed login attempts",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
        )

        recent_credential_change = st.selectbox(
            "Recent account credential change?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with right:
        transaction_amount = st.number_input(
            "Transaction amount (₹)",
            min_value=10.0,
            max_value=200000.0,
            value=8000.0,
            step=100.0,
        )

        average_transaction_amount = st.number_input(
            "Average historical transaction amount (₹)",
            min_value=10.0,
            max_value=100000.0,
            value=1500.0,
            step=100.0,
        )

        transaction_count_24h = st.number_input(
            "Transactions in the last 24 hours",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
        )

        unusual_location = st.selectbox(
            "Unusual transaction location?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    submitted = st.form_submit_button(
        "Analyze fraud risk",
        use_container_width=True,
    )


# --------------------------------------------------
# 5. Predict and display results
# --------------------------------------------------

if submitted:
    transaction = pd.DataFrame(
        [{
            "hours_since_sim_change": hours_since_sim_change,
            "sim_change_detected": sim_change_detected,
            "new_device": new_device,
            "failed_logins": failed_logins,
            "transaction_amount": transaction_amount,
            "average_transaction_amount": average_transaction_amount,
            "transaction_count_24h": transaction_count_24h,
            "unusual_location": unusual_location,
            "recent_credential_change": recent_credential_change,
        }],
        columns=features,
    )

    prediction = int(model.predict(transaction)[0])
    probabilities = model.predict_proba(transaction)[0]
    fraud_index = list(model.classes_).index(1)
    risk_score = float(probabilities[fraud_index]) * 100

    st.divider()
    st.header("Analysis result")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Model risk score",
        f"{risk_score:.1f}%",
    )

    metric2.metric(
        "Model classification",
        "High risk" if prediction == 1 else "Lower risk",
    )

    metric3.metric(
        "Decision threshold",
        f"{threshold * 100:.0f}%",
    )

    st.progress(int(round(risk_score)))

    if prediction == 1:
        st.error(
            "⚠️ The model classified this transaction as "
            "simulated fraud. Further investigation is recommended."
        )
    else:
        st.success(
            "The model classified this transaction as lower risk. "
            "This does not guarantee that it is legitimate."
        )

    # Explain the observable signals without claiming they caused
    # the prediction.
    signals = []

    if sim_change_detected and hours_since_sim_change <= 24:
        signals.append("A SIM change was detected within the last 24 hours.")

    if new_device:
        signals.append("The transaction uses a new device.")

    if failed_logins >= 3:
        signals.append("There are at least three failed login attempts.")

    if transaction_amount > 3 * average_transaction_amount:
        signals.append(
            "The transaction amount exceeds three times the historical average."
        )

    if transaction_count_24h >= 5:
        signals.append("There are at least five transactions in the last 24 hours.")

    if unusual_location:
        signals.append("The transaction is from an unusual location.")

    if recent_credential_change:
        signals.append("Account credentials were recently changed.")

    st.subheader("Signals to review")

    if signals:
        for signal in signals:
            st.markdown(f"- {signal}")
    else:
        st.write(
            "No predefined review signals were triggered. "
            "The model may still classify the transaction as suspicious."
        )

    st.caption(
        "The signals above are review prompts, not explanations of "
        "the model's internal reasoning. Risk scores depend on the "
        "synthetic training data and must not be used alone to block "
        "real transactions."
    )
