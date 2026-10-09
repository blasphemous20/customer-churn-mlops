# Customer Churn MLOps

A customer churn prediction project using scikit-learn, MLflow, and Streamlit. The model is trained on the Telco customer churn dataset, tracked and registered in MLflow, and exposed through an interactive Streamlit application.

## Project structure

```text
customer-churn-mlops/
├── app/
│   └── app.py                 # Streamlit customer churn prediction app
├── be/
│   └── api.py                 # FastAPI customer data query API
├── data/
│   └── telco_churn.csv        # Training and query dataset
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   ├── evaluate.py            # Train and evaluate a model locally
│   ├── load_model.py          # Load the registered MLflow model
│   ├── predict.py             # Shared model loading and prediction logic
│   ├── preprocessing.py       # Feature preparation and preprocessing
│   └── train.py               # Train, evaluate, and register a model
├── .python-version            # Python version used by uv (3.14)
├── Dockerfile                 # Container image for the Streamlit app
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Pip dependency list
└── uv.lock                    # Locked dependency versions
```

## Requirements

- Windows PowerShell
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.14 (uv can install it for you)

## Set up the environment

Run these commands from the project root. The checked-in `.python-version` selects Python 3.14, and `uv sync` creates `.venv` and installs the locked project dependencies.

```powershell
uv python install 3.14
uv sync
```

You can run project commands with `uv run` without activating the virtual environment. To activate it in PowerShell instead:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run locally

### 1. Start the MLflow tracking server

In a PowerShell terminal at the project root, start MLflow with the project's local SQLite database. Binding to `0.0.0.0` makes it reachable from the app container; use this for local development only and apply appropriate firewall rules.

```powershell
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Open the MLflow interface at [http://127.0.0.1:5000](http://127.0.0.1:5000). Keep this terminal running while training or using the app.

### 2. Train and register the model

In a second PowerShell terminal, point MLflow clients at the server and train the model. Training creates an experiment run and registers `customer-churn-classifier` in the MLflow Model Registry.

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uv run python src/train.py
```

The Streamlit predictor loads model version 1, so train and register the model before starting the app.

### 3. Launch the Streamlit app

In a third PowerShell terminal:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uv run streamlit run app/app.py
```

Open the app at **[http://localhost:8501](http://localhost:8501)**. Enter customer details or load a high-risk/low-risk example, then select **Predict churn**.

### Optional: run the customer data API

From another terminal at the project root:

```powershell
uv run uvicorn be.api:app --reload --port 8001
```

The API documentation is available at [http://localhost:8001/docs](http://localhost:8001/docs).

## Run the Streamlit app in Docker

Start MLflow and train/register the model as described above, then build the image from the project root:

```powershell
docker build -t customer-churn-mlops .
```

Run the app container and connect it to the MLflow server on the host:

```powershell
docker run --rm -p 8501:8501 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  customer-churn-mlops
```

Open [http://localhost:8501](http://localhost:8501). On Linux, if `host.docker.internal` is not available, add `--add-host=host.docker.internal:host-gateway` to `docker run`. The container includes the app, source code, and dataset; the MLflow server and registered model remain external.

## Model and data notes

- The tracking server and project scripts use the project-root `mlflow.db` by default. MLflow artifacts are stored locally in `mlartifacts/`.
- `MLFLOW_TRACKING_URI` is set per PowerShell terminal above; set it again in any new terminal that runs a training or inference command.
- `src/evaluate.py` trains and evaluates locally without registering an MLflow model. Use `src/train.py` for the registered model consumed by the app.
- The local MLflow database (`mlflow.db`), legacy run store (`mlruns/`), and virtual environment (`.venv/`) are excluded from version control. MLflow stores artifacts in the generated `mlartifacts/` directory.
