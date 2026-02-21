# Airbnb Price Prediction API
## Overview
This project implements a **production-ready machine learning inference service** for predicting Airbnb rental prices in Sydney.

The system covers the full ML lifecycle:
* Data preprocessing and feature engineering
* Model training and cross-validation
* Model selection (Random Forest vs XGBoost)
* Feature importance analysis
* Model serialisation
* FastAPI-based inference service
* Dockerised deployment

The final result is a containerised ML microservice that can serve real-time rental price predictions.

## Problem Statement
Given listing-level features such as:
* Property type
* Room type
* Location (neighbourhood)
* Number of bedrooms and bathrooms
* Review scores
* Availability

We aim to predict the expected nightly rental price.

## Dataset
Sydney Airbnb Open Data (https://www.kaggle.com/datasets/tylerx/sydney-airbnb-open-data)

After cleaning:
* Around 36k listings
* Mixed numerical and categorical features
* High-cardinality categorical variables (e.g. neighbourhood)

## Feature Engineering
Key preprocessing steps:
* Removed extreme outliers in price
* Log transformation of target variable (`log1p(price)`)
* Selection of relevant numerical and categorical features
* One-hot encoding for categorical variables
* Training-serving consistency ensured via `sklearn.pipeline.Pipeline`

This avoids training-serving skew and ensures inference consistency.

## Model Training and Selection
Two models were compared:
* Random Forest Regressor
* XGBoost Regressor

Evaluation strategy:
* 80/20 hold-out split
* 5-fold cross-validation

Final performance:
* Hold-out RMSE (log scale): 0.0.4073
* 5-fold CV RMSE (log scale): 0.4127

XGBoost was selected as the final model based on cross-validation performance.

## Global Feature Importance
Feature importance analysis was performed using the trained XGBoost model.

Top contributing features:
* `room_type_Entire home/apt`
* `bedrooms`
* `room_type_Private room`
* Selected high-value neighbourhoods

This indicates that property type and room configuration are dominant pricing factors, while location contributes in a distributed manner across suburbs.

The feature importance plot is stored at:
```Bash
model/feature_importance.png
```

## System Architecture
```text
Client -> FastAPI -> Pydantic Validation -> Sklearn Pipeline -> Model -> Prediction
```
Key design decisions:
* Full preprocessing + model wrapped in a single pipeline
* Model loaded once at startup
* Strict request/response schema via Pydantic
* Health check endpoint for service monitoring
* Docker containerisation for reproducibility

## API Endpoints
### Health Check
```text
GET /health
```
Returns service status.

### Predict Rental Price
```text
POST /predict
```
Example request:
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
Response:
```json
{
  "predicted_price": 137.08
}
```
The API returns the predicted nightly price in original scale.

## Project Structure
```text
airbnb-price-service/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
│
├── model/
│   ├── train.py
│   ├── preprocess.py
│   ├── model.pkl
│   └── feature_importance.png
│
├── notebooks/
│   └── eda.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md
```
## Running Locally
```Bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open:
```Bash
http://127.0.0.1:8000/docs
```

## Running with Docker
Build image:
```Bash
docker build -t airbnb-price-api .
```
Run container:
```Bash
docker run -p 8000:8000 airbnb-price-api
```
Access:
```Bash
http://localhost:8000/docs
```

## Engineering Highlights
* Training-serving consistency via `sklearn.pipeline.Pipeline`
* Strict I/O validation using Pydantic
* Model selection based on cross-validation
* Containerised deployment
* Feature importance analysis for interpretability

## Future Improvements
* SHAP-based local interpretability
* Model performance monitoring
* Request logging and structured tracing
* CI/CD integration
* Caching layer for repeated predictions