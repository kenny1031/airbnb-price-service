import pandas as pd
from model.preprocess import clean_and_engineer, split_xy

def test_preprocess_pipeline():
    df = pd.DataFrame({
        "accommodates": [2],
        "bathrooms": [1],
        "bedrooms": [1],
        "beds": [1],
        "minimum_nights": [2],
        "availability_365": [100],
        "number_of_reviews": [5],
        "review_scores_rating": [90],
        "reviews_per_month": [1.2],
        "room_type": ["Entire home/apt"],
        "property_type": ["Apartment"],
        "neighbourhood_cleansed": ["Sydney"],
        "cancellation_policy": ["moderate"],
        "instant_bookable": ["t"],
        "price": [200]
    })

    df = clean_and_engineer(df)
    X, y = split_xy(df)

    assert X.shape[0] == 1
    assert y.shape[0] == 1