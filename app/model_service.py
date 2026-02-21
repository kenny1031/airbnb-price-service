"""
This file is for:
- Load the model
- Prediction
- Transforms log back to the original
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

pipeline = joblib.load(MODEL_PATH)

def predict_price(data: dict) -> float:
    df = pd.DataFrame([data])
    log_pred = pipeline.predict(df)[0]
    price = np.expm1(log_pred)
    return float(round(price, 2))