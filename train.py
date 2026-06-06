import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = "data/Churn_Modelling.csv"

CANDIDATE_DIR = "models/candidate"
CANDIDATE_MODEL_PATH = os.path.join(CANDIDATE_DIR, "churn_model.pkl")
CANDIDATE_METRICS_PATH = os.path.join(CANDIDATE_DIR, "metrics.json")

THRESHOLD = 0.35
RANDOM_STATE = 42


def build_preprocessor(X):
    cat_features = ["Geography", "Gender"]
    num_features = [col for col in X.columns if col not in cat_features]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


def evaluate_model(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1_score": float(f1_score(y_test, pred)),
        "threshold": float(THRESHOLD),
    }


def train_and_evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["model_type"] = model_name

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("threshold", THRESHOLD)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print(f"\nModel: {model_name}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
    }


def main():
    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    mlflow.set_experiment("customer_churn_prediction")

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for model_name, model in models.items():
        result = train_and_evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    best_result = max(
        results,
        key=lambda item: item["metrics"]["f1_score"],
    )

    best_model_name = best_result["model_name"]
    best_pipeline = best_result["pipeline"]
    best_metrics = best_result["metrics"]

    joblib.dump(best_pipeline, CANDIDATE_MODEL_PATH)

    with open(CANDIDATE_METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4, ensure_ascii=False)

    print("\nBest model selected:")
    print(f"Model: {best_model_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1-score: {best_metrics['f1_score']:.4f}")
    print(f"Candidate model saved to {CANDIDATE_MODEL_PATH}")
    print(f"Candidate metrics saved to {CANDIDATE_METRICS_PATH}")


if __name__ == "__main__":
    main()