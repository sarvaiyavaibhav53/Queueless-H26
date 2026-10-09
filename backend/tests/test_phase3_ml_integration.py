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
    # Phase 1 Correction 2 regression test: token creation MUST NOT invoke legacy predict_wait_time
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time") as mock_pred:
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == TokenStatus.WAITING.value
        assert data["predicted_wait_minutes"] is None
        mock_pred.assert_not_called()


def test_ml_failure_graceful_degradation(setup_data):
    # When joining queue, legacy predictor is not invoked and token is successfully created,
    # even if legacy predictor is patched to raise an error
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", side_effect=PredictionServiceError("ML is down")) as mock_pred:
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == TokenStatus.WAITING.value
        assert data["predicted_wait_minutes"] is None
        # Verify token exists and is valid
        assert data["token_number"] is not None
        assert data["queue_position"] > 0
        mock_pred.assert_not_called()


def test_invalid_prediction(setup_data):
    # Regression: Queue-join endpoint must not call legacy predictor even if predictor returns negative or NaN
    for invalid_val in [-5.0, math.nan]:
        with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time", return_value=invalid_val) as mock_pred:
            resp = client.post(
                f"/api/v1/queues/{setup_data['service'].id}/join",
                json={"office_id": setup_data['office'].id}
            )
            assert resp.status_code == 201
            assert resp.json()["predicted_wait_minutes"] is None
            mock_pred.assert_not_called()


def test_multiple_queue_joins(setup_data):
    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time") as mock_pred:
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
        assert data1["predicted_wait_minutes"] is None
        assert data2["predicted_wait_minutes"] is None
        mock_pred.assert_not_called()


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
