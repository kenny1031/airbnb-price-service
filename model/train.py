from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
from xgboost import XGBRegressor

from preprocess import clean_and_engineer, split_xy, NUM_FEATURES, CAT_FEATURES


DATA_PATH = "../data/listings_dec18.csv"
MODEL_PATH = "model.pkl"

def build_pipelines() -> Dict[str, Pipeline]:
    """Build full preprocessing + model pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    xgboost = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    pipelines = {}

    pipeline_xgb = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("xgboost", xgboost),
        ]
    )

    pipeline_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf", rf),
        ]
    )

    # Save pipelines to dictionary
    pipelines["xgboost"] = pipeline_xgb
    pipelines["rf"] = pipeline_rf

    return pipelines

def evaluate_holdout(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Train-test split evaluation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Hold-out RMSE (log scale): {rmse:.4f}")
    return rmse

def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """5-fold cross validation"""
    scores = cross_val_score(pipeline, X, y, cv=5,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_rmse = -scores.mean()
    print(f"CV RMSE (log scale): {cv_rmse:.4f}")
    return cv_rmse

def save_model(pipeline: Pipeline) -> None:
    """Save trained pipeline"""
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def main():
    # Load raw data
    df = pd.read_csv(DATA_PATH)

    # Clean and feature engineering
    df = clean_and_engineer(df)

    # Split features and target
    X, y = split_xy(df)

    # Build pipelines
    pipelines = build_pipelines()

    # Evaluate and select model
    optimal_cv_rmse = float("inf")
    optimal_pipeline = None
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        holdout_rmse = evaluate_holdout(pipeline, X, y)
        cv_rmse = evaluate_cv(pipeline, X, y)
        if cv_rmse < optimal_cv_rmse:
            optimal_cv_rmse = cv_rmse
            optimal_pipeline = pipeline

    print(f"\nSelected model with CV RMSE: {optimal_cv_rmse:.4f}")

    # Fit on full dataset and save
    optimal_pipeline.fit(X, y)
    save_model(optimal_pipeline)

if __name__ == "__main__":
    main()