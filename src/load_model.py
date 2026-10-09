# Mengimpor MLflow untuk mengambil model dari Model Registry
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
import sklearn

# Menggunakan registry yang sama untuk training dan inference.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
mlflow.set_tracking_uri(TRACKING_URI)

# Menentukan nama model yang sudah terdaftar di MLflow
MODEL_NAME = "customer-churn-classifier"

# Menentukan versi model yang ingin digunakan
MODEL_VERSION = "1"


# Membuat URI untuk mengambil model Version 1
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


# Memuat model dari MLflow Model Registry
try:
    model = mlflow.sklearn.load_model(model_uri)
except MlflowException as exc:
    if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
        raise
    raise RuntimeError(
        f"Model '{MODEL_NAME}' version {MODEL_VERSION} is not registered in "
        f"'{mlflow.get_registry_uri()}'. Run `python src/train.py` first."
    ) from exc


# Menampilkan informasi bahwa model berhasil dimuat
print("Model berhasil dimuat dari MLflow.")

# Menampilkan tipe model
print("\nTipe model:")
print(type(model))