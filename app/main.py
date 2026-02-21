from fastapi import FastAPI
from app.schemas import RentalRequest
from app.model_service import predict_price

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Airbnb Price Prediction API is running"}

@app.post("/predict")
def predict(request: RentalRequest):
    data = request.dict()
    predicted_price = predict_price(data)
    return {"predicted_price": predicted_price}