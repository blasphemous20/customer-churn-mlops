"""
Streamlit UI for the Telco Customer Churn Predictor.

Run from the project root:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import prediction
#from src.serving.inference import predict  # same inference pipeline the API uses

st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")

# --- Option lists (must match training data exactly) ---
YES_NO = ["Yes", "No"]
YES_NO_NOPHONE = ["Yes", "No", "No phone service"]
YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# --- Default values + example presets ---
DEFAULTS = {
    "SeniorCitizen": "No",
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1,
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0,
}

HIGH_RISK = {**DEFAULTS, "gender": "Female"}

LOW_RISK = {
    "SeniorCitizen": "No",
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "Yes",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "tenure": 60,
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0,
}

# Initialise widget state once
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_preset(preset: dict):
    """Callback: overwrite widget state before the next render."""
    for k, v in preset.items():
        st.session_state[k] = v


# --- Header ---
st.title("🔮 Telco Customer Churn Predictor")
st.markdown(
    "Fill in the customer details to get a churn prediction. "
    "The model identifies customers at risk of churning."
)
st.caption(
    "💡 Month-to-month contracts with fiber optic internet and electronic check "
    "payments tend to have higher churn rates."
)

# --- Example presets ---
c1, c2, _ = st.columns([1, 1, 4])
c1.button("Load high-risk example", on_click=load_preset, args=(HIGH_RISK,))
c2.button("Load low-risk example", on_click=load_preset, args=(LOW_RISK,))

# --- Input form ---
with st.form("churn_form"):
    st.subheader("Demographics")
    d1, d2, d3, d4 = st.columns(4)
    gender = d1.selectbox("Gender", ["Male", "Female"], key="gender")
    SeniorCitizen = d2.selectbox("Senior Citizen", YES_NO, key="SeniorCitizen")
    Partner = d3.selectbox("Partner", YES_NO, key="Partner")
    Dependents = d4.selectbox("Dependents", YES_NO, key="Dependents")

    st.subheader("Phone services")
    p1, p2 = st.columns(2)
    PhoneService = p1.selectbox("Phone Service", YES_NO, key="PhoneService")
    MultipleLines = p2.selectbox("Multiple Lines", YES_NO_NOPHONE, key="MultipleLines")

    st.subheader("Internet services")
    InternetService = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"], key="InternetService"
    )
    i1, i2, i3 = st.columns(3)
    OnlineSecurity = i1.selectbox("Online Security", YES_NO_NOINTERNET, key="OnlineSecurity")
    OnlineBackup = i2.selectbox("Online Backup", YES_NO_NOINTERNET, key="OnlineBackup")
    DeviceProtection = i3.selectbox("Device Protection", YES_NO_NOINTERNET, key="DeviceProtection")
    i4, i5, i6 = st.columns(3)
    TechSupport = i4.selectbox("Tech Support", YES_NO_NOINTERNET, key="TechSupport")
    StreamingTV = i5.selectbox("Streaming TV", YES_NO_NOINTERNET, key="StreamingTV")
    StreamingMovies = i6.selectbox("Streaming Movies", YES_NO_NOINTERNET, key="StreamingMovies")

    st.subheader("Contract & billing")
    b1, b2, b3 = st.columns(3)
    Contract = b1.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"], key="Contract"
    )
    PaperlessBilling = b2.selectbox("Paperless Billing", YES_NO, key="PaperlessBilling")
    PaymentMethod = b3.selectbox("Payment Method", PAYMENT_METHODS, key="PaymentMethod")

    st.subheader("Charges")
    n1, n2, n3 = st.columns(3)
    tenure = n1.number_input(
        "Tenure (months)", min_value=0, max_value=100, step=1, key="tenure"
    )
    MonthlyCharges = n2.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, step=1.0, key="MonthlyCharges"
    )
    TotalCharges = n3.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, step=10.0, key="TotalCharges"
    )

    submitted = st.form_submit_button("Predict churn", type="primary")

# --- Prediction ---
if submitted:
    payload = {
        "SeniorCitizen": SeniorCitizen,
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        with st.spinner("Predicting..."):
            result = str(prediction(payload))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        if "not likely" in result.lower():
            st.success(f"✅ {result}")
        else:
            st.warning(f"⚠️ {result}")

        with st.expander("Input sent to the model"):
            st.json(payload)