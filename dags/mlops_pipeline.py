from __future__ import annotations

import os
import json
import pickle
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

# -----------------------------
# Config (edit paths as you like)
# -----------------------------
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
BASE_DIR = os.path.join(AIRFLOW_HOME, "mlops_demo")
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REGISTRY_DIR = os.path.join(BASE_DIR, "registry")   # "promoted" models
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)

# Local MLflow file store (no server required)
MLFLOW_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("airflow-mlops-demo")

# Simple dataset: UCI Heart Disease (cleaned subset via a raw GitHub URL)
DATA_URL = "https://raw.githubusercontent.com/ghd7262/Heart-Attack-Prediction/master/heart.csv"

# Promotion rule
ACCURACY_THRESHOLD = 0.80

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="mlops_end_to_end_demo",
    description="Ingest -> Validate -> Train -> Evaluate -> Promote (MLOps demo with MLflow logging)",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "demo"],
) as dag:

    @task
    def ingest_data() -> str:
        """Download CSV and store locally; return local path."""
        local_path = os.path.join(DATA_DIR, "heart.csv")
        df = pd.read_csv(DATA_URL)
        df.to_csv(local_path, index=False)
        return local_path

    @task
    def validate_data(csv_path: str) -> str:
        """Basic schema + NA checks; store a validation report."""
        df = pd.read_csv(csv_path)

        # Minimal expectations for demo purposes
        required_cols = {"age", "sex", "cp", "trestbps", "chol", "thalach", "target"}
        missing = required_cols - set(df.columns)
        report = {
            "rows": len(df),
            "cols": list(df.columns),
            "missing_required": list(missing),
            "null_counts": df.isnull().sum().to_dict(),
        }

        report_path = os.path.join(ARTIFACTS_DIR, "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Quick sanity checks
        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains null values, please clean first.")

        return report_path

    @task
    def split_data(csv_path: str) -> dict:
        """Split into train/test and write to disk; return file paths."""
        df = pd.read_csv(csv_path)

        # Simple feature/label selection for demo
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        paths = {
            "X_train": os.path.join(ARTIFACTS_DIR, "X_train.parquet"),
            "X_test": os.path.join(ARTIFACTS_DIR, "X_test.parquet"),
            "y_train": os.path.join(ARTIFACTS_DIR, "y_train.parquet"),
            "y_test": os.path.join(ARTIFACTS_DIR, "y_test.parquet"),
        }
        X_train.to_parquet(paths["X_train"])
        X_test.to_parquet(paths["X_test"])
        pd.DataFrame({"target": y_train}).to_parquet(paths["y_train"])
        pd.DataFrame({"target": y_test}).to_parquet(paths["y_test"])

        return paths

    @task
    def train_model(paths: dict) -> str:
        """Train a logistic regression; log to MLflow; return model path."""
        X_train = pd.read_parquet(paths["X_train"])
        y_train = pd.read_parquet(paths["y_train"])["target"]

        with mlflow.start_run(run_name="logreg-train") as run:
            params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs"}
            mlflow.log_params(params)

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Log the serialized model as an artifact
            mlflow.log_artifact(model_path, artifact_path="model")

            # Save run info for downstream tasks
            meta_path = os.path.join(ARTIFACTS_DIR, "mlflow_run.json")
            with open(meta_path, "w") as f:
                json.dump({"run_id": run.info.run_id}, f)

        return model_path

    @task
    def evaluate_model(paths: dict, model_path: str) -> dict:
        """Evaluate on test; log metrics to MLflow; return metrics dict."""
        X_test = pd.read_parquet(paths["X_test"])
        y_test = pd.read_parquet(paths["y_test"])["target"]

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metric to the most recent run (by reading the run_id we wrote)
        meta_path = os.path.join(ARTIFACTS_DIR, "mlflow_run.json")
        run_id = json.load(open(meta_path))["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", float(acc))

        metrics = {"accuracy": float(acc), "threshold": ACCURACY_THRESHOLD}
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2)
        return metrics

    @task
    def promote_if_good(model_path: str, metrics: dict) -> str:
        """
        If model meets the accuracy threshold, "promote" it by copying into a
        simple filesystem registry with a versioned filename. Otherwise, skip.
        """
        acc = metrics["accuracy"]
        if acc < ACCURACY_THRESHOLD:
            # Skipping is nice to visualize in the UI as a yellow task
            raise AirflowSkipException(
                f"Accuracy {acc:.3f} < threshold {ACCURACY_THRESHOLD:.3f}. Not promoting."
            )

        version_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = os.path.join(REGISTRY_DIR, f"model_{version_tag}_acc{acc:.3f}.pkl")
        with open(model_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        return dest

    csv_path = ingest_data()
    _report = validate_data(csv_path)
    splits = split_data(csv_path)
    model_path = train_model(splits)
    metrics = evaluate_model(splits, model_path)
    _maybe_promoted = promote_if_good(model_path, metrics)
