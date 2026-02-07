from fastapi import FastAPI
import pandas as pd

from app.schema import IVFInput
from app.load_model import model

app = FastAPI(title="IVF Trigger Day Prediction API")

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: IVFInput):
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_alias": "production"
    }