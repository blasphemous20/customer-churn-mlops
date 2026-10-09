# Mengimpor pandas untuk membaca dataset
import os
from pathlib import Path

import pandas as pd

# Mengimpor MLflow untuk mencatat eksperimen
import mlflow
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
mlflow.set_tracking_uri(TRACKING_URI)

# Mengimpor fungsi untuk membagi dataset
from sklearn.model_selection import train_test_split

# Mengimpor Logistic Regression sebagai baseline model
from sklearn.linear_model import LogisticRegression

# Mengimpor Pipeline untuk menggabungkan preprocessing dan model
from sklearn.pipeline import Pipeline

# Mengimpor metrik evaluasi
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Mengimpor fungsi preprocessing yang sudah dibuat
from preprocessing import prepare_dataframe, create_preprocessor


# Menentukan lokasi dataset
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"


# Membaca dataset
df = pd.read_csv(DATA_PATH)

# Menyiapkan fitur dan target
X, y = prepare_dataframe(df)


# Membagi dataset menjadi data training dan testing
# Stratify menjaga proporsi kelas Churn tetap sama
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Membuat preprocessing
preprocessor = create_preprocessor()


# Menentukan parameter model
MAX_ITER = 1000


# Membuat Logistic Regression
model = LogisticRegression(
    max_iter=MAX_ITER,
    random_state=42
)


# Menggabungkan preprocessing dan model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Menentukan nama experiment MLflow
mlflow.set_experiment("customer-churn")


# Memulai pencatatan experiment
with mlflow.start_run():

    # Melatih model
    model_pipeline.fit(X_train, y_train)

    # Membuat prediksi kelas
    y_pred = model_pipeline.predict(X_test)

    # Menghasilkan probabilitas Churn
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Mencatat parameter model ke MLflow
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", MAX_ITER)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Mencatat metrics ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Menyimpan model pipeline sekaligus mendaftarkannya ke Model Registry
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        name="customer_churn_model",
        skops_trusted_types=["numpy.dtype"],
        registered_model_name="customer-churn-classifier"
    )

    # Menampilkan hasil evaluasi di terminal
    print("Training berhasil.")
    print("\nHasil evaluasi:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Menampilkan Run ID MLflow
    print("\nMLflow Run ID:")
    print(mlflow.active_run().info.run_id)