from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint_returns_price(monkeypatch):
    # fake pipeline
    class DummyPipeline:
        def predict(self, X):
            return [5.0]

    monkeypatch.setattr("app.model_service.pipeline", DummyPipeline())

    with TestClient(app) as client:
        response = client.post("/predict", json={
            "accommodates": 2,
            "bathrooms": 1,
            "bedrooms": 1,
            "beds": 1,
            "minimum_nights": 2,
            "availability_365": 200,
            "number_of_reviews": 10,
            "review_scores_rating": 90,
            "reviews_per_month": 1.2,
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "neighbourhood_cleansed": "Sydney",
            "cancellation_policy": "moderate",
            "instant_bookable": "t"
        })

    assert response.status_code == 200