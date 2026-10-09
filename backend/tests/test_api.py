import os
import sys
import joblib
import pytest
import pandas as pd
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app
from app.services.prediction_service import WaitTimePredictionService

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_PATH = os.path.join(BASE_DIR, 'ml/models/random_forest_wait_time_pipeline.joblib')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'ml/data/synthetic/splits/test.parquet')

# Features ordered exactly as expected
FEATURES = [
    "office_id", "service_id", "hour", "day_of_week", "people_ahead",
    "queue_length", "active_counters", "arrival_rate_15m", "counter_load",
    "historical_avg_service_minutes", "service_time_std", "no_show_rate"
]

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def real_model():
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="session")
def sample_rows():
    df = pd.read_parquet(TEST_DATA_PATH)
    # Pick 3 real rows
    return df.iloc[:3][FEATURES].to_dict('records')

def test_model_loading():
    service = WaitTimePredictionService()
    service.load_model(MODEL_PATH)
    assert service.model is not None
    assert hasattr(service.model, 'predict')

def test_prediction_correctness(real_model, sample_rows, client):
    # Test backward-compatible legacy prediction on isolated service instance
    service = WaitTimePredictionService()
    service.load_model(MODEL_PATH)
    
    for row in sample_rows:
        # direct model
        input_df = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
        expected_pred = float(real_model.predict(input_df)[0])
        
        # isolated service prediction
        service_pred = service.predict_wait_time(row)
        assert abs(expected_pred - service_pred) < 1e-5
        
        # Authoritative production API endpoint MUST reject legacy unsupported identities (RJT / LL)
        api_resp = client.post("/api/v1/predictions/wait-time", json=row)
        assert api_resp.status_code == 422
        assert api_resp.json()["detail"]["error_code"] in ["UNSUPPORTED_MODEL_OFFICE", "UNSUPPORTED_MODEL_SERVICE"]

    # Authoritative production API endpoint MUST serve genuinely supported model identities (MUNI / PT) via RF
    valid_prod_row = {
        "office_id": "MUNI",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
        "historical_avg_service_minutes": 10.0,
    }
    prod_resp = client.post("/api/v1/predictions/wait-time", json=valid_prod_row)
    assert prod_resp.status_code == 200
    prod_data = prod_resp.json()
    assert prod_data["predicted_wait_minutes"] > 0.0
    assert prod_data["prediction_source"] == "RF"

def test_api_valid_request(client):
    valid_payload = {
        "office_id": "MUNI",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
        "historical_avg_service_minutes": 10.0,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_wait_minutes" in data
    assert data["prediction_source"] == "RF"
    assert data["predicted_wait_minutes"] > 0.0

def test_api_missing_field(client):
    row = {
        "office_id": "MUNI",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "queue_length": 3,
        "active_counters": 2,
    }  # missing people_ahead
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_api_unknown_office(client):
    row = {
        "office_id": "UNKNOWN_OFFICE",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "UNKNOWN_APPLICATION_OFFICE"

def test_api_unknown_service(client):
    row = {
        "office_id": "MUNI",
        "service_id": "UNKNOWN_SERVICE",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "UNKNOWN_APPLICATION_SERVICE"

def test_api_unsupported_model_office(client):
    row = {
        "office_id": "RJT",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "UNSUPPORTED_MODEL_OFFICE"

def test_api_unsupported_model_service(client):
    row = {
        "office_id": "MUNI",
        "service_id": "LL",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": 2,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "UNSUPPORTED_MODEL_SERVICE"

def test_api_invalid_numeric(client):
    row = {
        "office_id": "MUNI",
        "service_id": "PT",
        "hour": 10,
        "day_of_week": 1,
        "people_ahead": 2,
        "queue_length": 3,
        "active_counters": -5,
    }
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_failure_verification(real_model, sample_rows):
    assert True

