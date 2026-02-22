import os
import joblib
from sqlalchemy import create_engine, text

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(db_url)


def load_latest_model():
    engine = get_engine()

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT file_path
                FROM model_versions
                ORDER BY created_at DESC
                LIMIT 1
            """)
        ).fetchone()

    if result is None:
        raise RuntimeError("No model found in registry")

    model_path = result[0]

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model