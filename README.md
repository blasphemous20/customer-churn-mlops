# Customer Churn MLOps

Machine Learning Operations project for customer churn prediction.

## Team

- Fajar: Machine Learning & MLOps
- [Nama Teman]&#58; API, Deployment & CI/CD

## Tech Stack

- Python
- Scikit-learn
- MLflow
- FastAPI
- Docker
- GitHub Actions

## Train and load the model

Train and register the model before loading it:

```bash
python src/train.py
python src/load_model.py
```

By default, MLflow uses the project-root `mlflow.db` registry. Set
`MLFLOW_TRACKING_URI` when using a different MLflow tracking server or database.