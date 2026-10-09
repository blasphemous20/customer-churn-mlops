# Mengimpor pandas untuk mengolah dataframe
import pandas as pd

# Mengimpor komponen preprocessing dari scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# Menentukan kolom numerik yang digunakan sebagai fitur model
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# Menentukan kolom kategorikal yang digunakan sebagai fitur model
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def prepare_dataframe(df):
    """
    Menyiapkan dataframe sebelum masuk ke proses preprocessing.
    """

    # Membuat salinan dataframe agar data asli tidak berubah
    data = df.copy()

    # Mengubah TotalCharges menjadi tipe numerik
    # Nilai kosong akan diubah menjadi NaN
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # Menghapus customerID karena hanya merupakan identitas pelanggan
    data = data.drop(columns=["customerID"])

    # Memisahkan fitur dari target
    X = data.drop(columns=["Churn"])

    # Mengubah target Churn menjadi nilai numerik
    y = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


def create_preprocessor():
    """
    Membuat pipeline preprocessing untuk fitur numerik dan kategorikal.
    """

    # Membuat pipeline untuk fitur numerik
    # Nilai kosong akan diganti menggunakan median
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Membuat pipeline untuk fitur kategorikal
    # Nilai kosong akan diganti menggunakan kategori yang paling sering muncul
    # Kemudian data kategorikal diubah menjadi angka menggunakan One-Hot Encoding
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Menggabungkan preprocessing numerik dan kategorikal
    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor