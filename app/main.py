import os

from fastapi import FastAPI, HTTPException
from sqlalchemy.engine import row

from app.schemas import RentalRequest, RentalResponse
from app.model_service import predict_price, load_model
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL")

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(db_url)

app = FastAPI(
    title="Airbnb Price Prediction API",
    version="2.0.0",
    description="Production-style ML service with model registry"
)

@app.on_event("startup")
def startup():
    """Startup event"""
    try:
        load_model()
        print("Model successfully loaded.")
    except Exception as e:
        print(f"Model loading failed: {e}")

@app.get("/")
def root():
    """Root"""
    return {"message": "Airbnb Price Prediction API is running"}

@app.get("/health")
def health_check():
    """Health Check"""
    return {"status": "healthy"}

@app.post("/predict", response_model=RentalResponse)
def predict(request: RentalRequest) -> RentalResponse:
    """Prediction Endpoint"""
    try:
        data = request.dict()
        predicted_price = predict_price(data)
        return RentalResponse(predicted_price=predicted_price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model():
    try:
        load_model()
        return {"status": "model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    try:
        #engine = get_engine()
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    SELECT version, rmse, created_at
                    FROM model_versions
                    ORDER BY created_at DESC
                """)
            ).fetchall()

        models = [
            {
                "version": row[0],
                "rmse": row[1],
                "created_at": row[2]
            }
            for row in result
        ]

        return {"models": models}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))