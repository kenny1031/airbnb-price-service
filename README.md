# Airbnb Price Prediction Service

A production-style machine learning API service for predicting Airbnb listing prices, built with FastAPI, PostgreSQL, Docker, and a versioned model registry.

## Key Features

- Versioned model registry stored in PostgreSQL
- Serialised sklearn pipeline with joblib
- FastAPI prediction endpoint
- Fully Dockerised (API + Postgres)
- Unit & integration tests with pytest
- Dependency injection for testable architecture

## Tech Stack

- Python 3.10
- FastAPI
- PostgreSQL
- SQLAlchemy
- Scikit-learn
- Docker & Docker Compose
- Pytest

## Project Structure
```text
airbnb-price-service/
│
├── app/                 # FastAPI application
│   ├── main.py          # API entry point
│   ├── schemas.py       # Pydantic request/response models
│   ├── model_service.py # Prediction logic
│   └── model_loader.py  # Model registry loader
│
├── model/
│   ├── train.py
│   └── preprocess.py
│
├── models/ # Saved model artifacts
│
├── db/ # Database schema
│   └── schema.sql
│
├── scripts/
│   └── load_data.py
│
├── tests/ # Unit & integration tests
│   ├── unit
│   │   ├── test_model_loader.py
│   │   └── test_preprocess.py   
│   └── integration
│       └── test_predict_api.py
│
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Architecture Overview

1. Model is trained using `model/train.py`
2. Trained model is serialized and stored on disk
3. Model metadata (version, rmse, file_path) is saved in PostgreSQL
4. API loads the latest model at startup
5. `/predict` endpoint returns predicted price

The system follows a clean separation of concerns with dependency injection for testability.

## Run with Docker
```bash
docker compose build
docker compose up
```
API available at:
```text
http://localhost:8000/docs
```

## Train a Model
```bash
docker compose run api python -m model.train
```

## Run Tests
```bash
python -m pytest -v
```
- Unit tests mock database engine
- Integration tests mock prediction pipeline

## Example Prediction Request
```json
{
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
  "cancellation_policy": "flexible",
  "instant_bookable": "t"
}
```

## Future Improvements

- Add CI/CD with GitHub Actions
- Add model performance tracking dashboard
- Support A/B model serving
- Implement model rollback endpoint