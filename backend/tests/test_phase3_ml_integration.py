import pytest
import math
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest import mock

from app.main import app
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter
from app.services.prediction_service import PredictionServiceError

client = TestClient(app)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    db.query(QueueToken).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    yield db
    db.query(QueueToken).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    db.close()

@pytest.fixture(scope="module")
def setup_data(db_session: Session):
    # Using specific names to avoid conflict if tests are run in parallel, though they run sequentially
    office = Office(code="RJT", name="Rajkot RTO", counters=3, demand_multiplier=1.0)
    service = Service(code="LL", name="Learning License", mean_minutes=15.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_successful_ml_integration(setup_data):
    # We mock prediction_service to return a specific float value
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", return_value=25.6):
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == TokenStatus.WAITING.value
        assert data["predicted_wait_minutes"] == 26  # rounded 25.6

def test_ml_failure_graceful_degradation(setup_data):
    # Mock to raise an exception
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", side_effect=PredictionServiceError("ML is down")):
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        # Should still succeed
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == TokenStatus.WAITING.value
        assert data["predicted_wait_minutes"] is None
        
        # Verify token exists and is valid
        assert data["token_number"] is not None
        assert data["queue_position"] > 0

def test_invalid_prediction(setup_data):
    # Test negative
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", return_value=-5.0):
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp.status_code == 201
        assert resp.json()["predicted_wait_minutes"] is None
        
    # Test non-finite
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", return_value=math.nan):
        resp2 = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp2.status_code == 201
        assert resp2.json()["predicted_wait_minutes"] is None

def test_multiple_queue_joins(setup_data):
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", return_value=12.0):
        resp1 = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        resp2 = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        
        data1 = resp1.json()
        data2 = resp2.json()
        
        assert data1["token_number"] != data2["token_number"]
        assert data1["queue_position"] < data2["queue_position"]
        assert data1["predicted_wait_minutes"] == 12
        assert data2["predicted_wait_minutes"] == 12

def test_existing_queue_transitions(setup_data):
    # Call-next to process one of the tokens
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 200
    token_id = resp.json()["id"]
    
    # Start
    start_resp = client.post(f"/api/v1/tokens/{token_id}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == TokenStatus.SERVING.value
    
    # Complete
    comp_resp = client.post(f"/api/v1/tokens/{token_id}/complete")
    assert comp_resp.status_code == 200
    assert comp_resp.json()["status"] == TokenStatus.COMPLETED.value
