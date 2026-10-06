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
    service = WaitTimePredictionService()
    service.load_model(MODEL_PATH)
    
    for row in sample_rows:
        # direct model
        input_df = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
        expected_pred = float(real_model.predict(input_df)[0])
        
        # service prediction
        service_pred = service.predict_wait_time(row)
        
        # API prediction
        api_resp = client.post("/api/v1/predictions/wait-time", json=row)
        assert api_resp.status_code == 200
        api_pred = api_resp.json()['predicted_wait_minutes']
        
        print(f"Direct: {expected_pred}, Service: {service_pred}, API: {api_pred}")
        
        # INTENTIONAL FAILURE REVERTED
        assert abs(expected_pred - service_pred) < 1e-5
        assert abs(expected_pred - api_pred) < 1e-5

def test_api_valid_request(sample_rows, client):
    row = sample_rows[0]
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 200
    assert "predicted_wait_minutes" in resp.json()

def test_api_missing_field(sample_rows, client):
    row = sample_rows[0].copy()
    del row["people_ahead"]
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_api_invalid_office(sample_rows, client):
    row = sample_rows[0].copy()
    row["office_id"] = "INVALID"
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_api_invalid_service(sample_rows, client):
    row = sample_rows[0].copy()
    row["service_id"] = "INVALID"
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_api_invalid_numeric(sample_rows, client):
    row = sample_rows[0].copy()
    row["active_counters"] = -5
    resp = client.post("/api/v1/predictions/wait-time", json=row)
    assert resp.status_code == 422

def test_failure_verification(real_model, sample_rows):
    # This test will be temporarily edited to fail during verification
    assert True
