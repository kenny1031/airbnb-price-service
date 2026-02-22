from typing import Dict
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# DB
from sqlalchemy import create_engine, text

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from preprocess import clean_and_engineer, split_xy, NUM_FEATURES, CAT_FEATURES

# Database Connection
DB_URL = os.getenv("DATABASE_URL")
if DB_URL is None:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DB_URL)

def build_pipelines() -> Dict[str, Pipeline]:
    """Build Pipelines"""
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

    xgb = XGBRegressor(
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
        random_state=42,
        n_jobs=-1,
    )

    return {
        "xgboost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", rf),
        ]),
    }


def evaluate_holdout(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Hold-out RMSE: {rmse:.4f}")
    return rmse


def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> float:
    scores = cross_val_score(pipeline, X, y,cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_rmse = -scores.mean()
    print(f"CV RMSE: {cv_rmse:.4f}")
    return cv_rmse

def generate_feature_importance(pipeline: Pipeline) -> None:
    """Main Training Logic"""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return

    num_features = NUM_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat = cat_encoder.get_feature_names_out(CAT_FEATURES)

    feature_names = list(num_features) + list(encoded_cat)
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(
        np.array(feature_names)[indices][::-1],
        importances[indices][::-1]
    )
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

def main():
    """Main Training Logic"""
    print("Loading data from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM listings", engine)

    df = clean_and_engineer(df)
    X, y = split_xy(df)

    pipelines = build_pipelines()

    best_cv_rmse = float("inf")
    best_pipeline = None

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        evaluate_holdout(pipeline, X, y)
        cv_rmse = evaluate_cv(pipeline, X, y)

        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_pipeline = pipeline

    print(f"\nSelected model with CV RMSE: {best_cv_rmse:.4f}")

    # Fit on full dataset
    best_pipeline.fit(X, y)

    # Create model version
    version = f"model_{uuid.uuid4().hex[:8]}"
    model_path = f"models/{version}.pkl"

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    # Save model metadata to DB
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO model_versions (version, rmse, file_path)
                VALUES (:version, :rmse, :file_path)
            """),
            {
                "version": version,
                "rmse": float(best_cv_rmse),
                "file_path": model_path,
            }
        )

    generate_feature_importance(best_pipeline)

    print(f"Model saved as {model_path}")
    print("Model metadata recorded in database.")

if __name__ == "__main__":
    main()