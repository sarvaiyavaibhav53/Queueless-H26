import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.database import get_db, SessionLocal, engine
from app.models.office import Office
from app.main import app

client = TestClient(app)

def test_database_connection():
    # Test SELECT 1
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar()
        assert result == 1

def test_db_health_endpoint():
    # Test the API health endpoint
    response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "Database connection successful"

def test_model_access():
    # Test basic session and model access
    db = SessionLocal()
    try:
        # Should not raise exception
        offices = db.query(Office).all()
        assert isinstance(offices, list)
    finally:
        db.close()
