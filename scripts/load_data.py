import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from model.preprocess import NUM_FEATURES, CAT_FEATURES, TARGET_RAW
import os

# DB connection
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

# Load CSV
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "listings_dec18.csv"

df = pd.read_csv(DATA_PATH)

# Select only required columns
columns = ["id"] + NUM_FEATURES + CAT_FEATURES + [TARGET_RAW]
df = df[columns].copy()
df.rename(columns={"id": "listing_id"}, inplace=True)

# Clean price (remove $ if needed)
df[TARGET_RAW] = (
    df[TARGET_RAW]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Insert into DB
df.to_sql("listings", engine, if_exists="replace", index=False)
print("Data successfully loaded into PostgreSQL")
