import joblib
import pandas as pd

from fastapi import FastAPI


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

model = joblib.load("models/churn_model.pkl")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "churn_model"
    }


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= 0.35)

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": "1.0.0"
    }