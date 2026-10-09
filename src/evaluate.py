# Mengimpor pandas untuk membaca dataset
from mlflow import data
import pandas as pd

# Mengimpor fungsi untuk membagi dataset
from sklearn.model_selection import train_test_split

# Mengimpor Logistic Regression sebagai model baseline
from sklearn.linear_model import LogisticRegression

# Mengimpor Pipeline untuk menggabungkan preprocessing dan model
from sklearn.pipeline import Pipeline

# Mengimpor metrik evaluasi
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Mengimpor fungsi preprocessing dari file preprocessing.py
from preprocessing import prepare_dataframe, create_preprocessor


# Menentukan lokasi dataset
DATA_PATH = "data/telco_churn.csv"


# Membaca dataset
df = pd.read_csv(DATA_PATH)

# Menyiapkan fitur dan target
X, y = prepare_dataframe(df)


# Membagi dataset menjadi training dan testing
# Stratify menjaga proporsi kelas Churn pada kedua dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Membuat preprocessing
preprocessor = create_preprocessor()


# Membuat Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Menggabungkan preprocessing dan model
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Melatih model menggunakan data training
model_pipeline.fit(X_train, y_train)


# Menghasilkan prediksi kelas
y_pred = model_pipeline.predict(X_test)

# Menghasilkan probabilitas prediksi kelas positif
# Kelas positif adalah Churn = 1
y_proba = model_pipeline.predict_proba(X_test)[:, 1]


# Menghitung accuracy
accuracy = accuracy_score(y_test, y_pred)

# Menghitung precision
precision = precision_score(y_test, y_pred)

# Menghitung recall
recall = recall_score(y_test, y_pred)

# Menghitung F1-score
f1 = f1_score(y_test, y_pred)

# Menghitung ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)


# Menampilkan hasil evaluasi
print("Hasil Evaluasi Model")
print("=" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# Menampilkan classification report
print("\nClassification Report")
print("=" * 30)
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# Menampilkan confusion matrix
print("Confusion Matrix")
print("=" * 30)
print(confusion_matrix(y_test, y_pred))