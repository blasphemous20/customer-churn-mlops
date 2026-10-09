"""Load the registered churn model and predict churn for one customer."""

from functools import lru_cache
import os
from pathlib import Path

import mlflow
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "customer-churn-classifier"
MODEL_VERSION = "1"

mlflow.set_experiment("Telco customer churn prediction")
mlflow.sklearn.autolog()

@lru_cache(maxsize=1)
def _load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    except Exception as e:
        print(f"Failed to create model URI: {e}")
    return mlflow.sklearn.load_model(model_uri)

def prediction(customer: dict) -> str:
    """Return a readable churn prediction for a customer's feature values."""
    customer_data = pd.DataFrame([customer])
    if "SeniorCitizen" in customer_data:
        customer_data["SeniorCitizen"] = customer_data["SeniorCitizen"].replace(
            {"No": 0, "Yes": 1}
        )

    model = _load_model()
    predicted_churn = model.predict(customer_data)[0] == 1
    probability = model.predict_proba(customer_data)[0][1]
    likelihood = "likely" if predicted_churn else "not likely"
    return f"Customer is {likelihood} to churn (probability: {probability:.1%})."