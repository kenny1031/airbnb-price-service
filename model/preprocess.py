from __future__ import annotations

import numpy as np
import pandas as pd

# Numerical features of interest
NUM_FEATURES = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
]

# Categorical features of interest
CAT_FEATURES = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "cancellation_policy",
    "instant_bookable",
]

# Target variable
TARGET_RAW = "price"
TARGET = "log_price"

def _to_float_money(s: pd.Series) -> pd.Series:
    """Converts money values from $xxx to float xxx"""
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

def clean_and_engineer(df: pd.DataFrame, max_price: float=2000.0) -> pd.DataFrame:
    """Clean and engineer the features"""
    df = df.copy()

    # Clean target
    df[TARGET_RAW] = _to_float_money(df[TARGET_RAW])

    # Remove extreme outliers
    df = df[df[TARGET_RAW] <= max_price]

    # Log-transform targt
    df[TARGET] = np.log1p(df[TARGET_RAW])

    # Drop tiny missing numeric cols
    df = df.dropna(subset=["bathrooms", "bedrooms", "beds", "beds"])

    # Fill review-related missing values with business logic
    df["review_scores_rating"] = df["review_scores_rating"].fillna(0)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Keep only required columns
    keep_cols = NUM_FEATURES + CAT_FEATURES + [TARGET]
    return df[keep_cols]

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into features matrix and labels vector"""
    X = df[NUM_FEATURES + CAT_FEATURES]
    try:
        y = df[TARGET]
    except KeyError:
        print("Target not found, ensure the data is cleaned before splitting!")
    return X, y