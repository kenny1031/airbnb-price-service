"""
This file is for:
- Load the model
- Prediction
- Transforms log back to the original
"""
import numpy as np
import pandas as pd
from app.model_loader import load_latest_model

pipeline = None

def load_model():
    global pipeline
    pipeline = load_latest_model()

# Load once when module imports
load_model()

def predict_price(data: dict) -> float:
    if pipeline is None:
        raise RuntimeError("Model not loaded")
    df = pd.DataFrame([data])
    log_pred = pipeline.predict(df)[0]
    price = np.expm1(log_pred)
    return float(round(price, 2))