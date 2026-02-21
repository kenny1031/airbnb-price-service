from fastapi import FastAPI, HTTPException
from app.schemas import RentalRequest, RentalResponse
from app.model_service import predict_price
from typing import Dict

app = FastAPI(
    title="Airbnb Price Prediction API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Airbnb Price Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=RentalResponse)
def predict(request: RentalRequest) -> RentalResponse:
    try:
        data = request.dict()
        predicted_price = predict_price(data)
        return RentalResponse(predicted_price=predicted_price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))