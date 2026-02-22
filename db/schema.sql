-- execute: psql -U kennyyu -h localhost -p 55432 -d airbnb_ml -f db/schema.sql
-- Listings Table
CREATE TABLE IF NOT EXISTS listings (
    listing_id BIGINT PRIMARY KEY,
    -- Numerical features
    accommodates INTEGER,
    bathrooms FLOAT,
    bedrooms FLOAT,
    beds FLOAT,
    minimum_nights INTEGER,
    availability_365 INTEGER,
    number_of_reviews INTEGER,
    review_scores_rating FLOAT,
    reviews_per_month FLOAT,
    -- Categorical features
    room_type VARCHAR(50),
    property_type VARCHAR(50),
    neighbourhood_cleansed VARCHAR(100),
    cancellation_policy VARCHAR(50),
    instant_bookable VARCHAR(5),
    -- Target (raw scale)
    price FLOAT,
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster neighbourhood queries
CREATE INDEX IF NOT EXISTS  idx_listings_neighbourhood
ON listings (neighbourhood_cleansed);

-- Model Versions Table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    rmse FLOAT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index to quickly fetch latest model
CREATE INDEX IF NOT EXISTS idx_model_versions_created_at
ON model_versions (created_at DESC);