import joblib
import pytest
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from app import model_loader

@pytest.fixture
def temp_db(tmp_path):
    """Create temporary SQLite DB for testing"""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                rmse FLOAT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

    return engine

@pytest.fixture
def dummy_model(tmp_path):
    """Create a temporary sklearn pipeline file."""
    model_path = tmp_path / "dummy.pkl"

    pipeline = Pipeline([
        ("model", DummyRegressor(strategy="mean"))
    ])

    joblib.dump(pipeline, model_path)
    return str(model_path)

def test_load_latest_model_success(temp_db, dummy_model, monkeypatch):
    """
    Should load latest model correctly.
    """

    # Insert fake model record
    with temp_db.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_versions (version, rmse, file_path)
            VALUES ('v1', 0.5, :file_path)
        """), {"file_path": dummy_model})

    # Override engine in model_loader
    monkeypatch.setattr(model_loader, "_engine", temp_db)

    pipeline = model_loader.load_latest_model()

    assert pipeline is not None
    assert hasattr(pipeline, "predict")

def test_load_latest_model_file_not_found(temp_db, monkeypatch):
    """
    Should raise RuntimeError if file missing.
    """

    with temp_db.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_versions (version, rmse, file_path)
            VALUES ('v1', 0.5, 'nonexistent.pkl')
        """))

    monkeypatch.setattr(model_loader, "_engine", temp_db)

    with pytest.raises(RuntimeError):
        model_loader.load_latest_model()