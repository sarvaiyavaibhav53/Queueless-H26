import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from app.main import app
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter

client = TestClient(app, raise_server_exceptions=True)

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
    office1 = Office(code="O1", name="Office 1", counters=3, demand_multiplier=1.0)
    office2 = Office(code="O2", name="Office 2", counters=2, demand_multiplier=1.0)
    service1 = Service(code="S1", name="Service 1", mean_minutes=10.0)
    service2 = Service(code="S2", name="Service 2", mean_minutes=15.0)
    db_session.add_all([office1, office2, service1, service2])
    db_session.commit()
    db_session.refresh(office1)
    db_session.refresh(office2)
    db_session.refresh(service1)
    db_session.refresh(service2)
    
    now = datetime.now(timezone.utc)
    
    tokens = [
        QueueToken(token_number="1", office_id=office1.id, service_id=service1.id, status=TokenStatus.WAITING, created_at=now, predicted_wait_minutes=5),
        QueueToken(token_number="2", office_id=office1.id, service_id=service1.id, status=TokenStatus.CALLED, created_at=now - timedelta(minutes=10), called_at=now, predicted_wait_minutes=10),
        QueueToken(token_number="3", office_id=office1.id, service_id=service1.id, status=TokenStatus.SERVING, created_at=now - timedelta(minutes=20), called_at=now - timedelta(minutes=10), serving_at=now - timedelta(minutes=5), predicted_wait_minutes=15),
        QueueToken(token_number="4", office_id=office1.id, service_id=service1.id, status=TokenStatus.COMPLETED, created_at=now - timedelta(minutes=30), called_at=now - timedelta(minutes=20), serving_at=now - timedelta(minutes=15), completed_at=now, predicted_wait_minutes=20),
        QueueToken(token_number="5", office_id=office2.id, service_id=service2.id, status=TokenStatus.CANCELLED, created_at=now, cancelled_at=now),
        QueueToken(token_number="6", office_id=office2.id, service_id=service2.id, status=TokenStatus.SKIPPED, created_at=now, called_at=now, skipped_at=now)
    ]
    db_session.add_all(tokens)
    db_session.commit()
    
    return {
        "office1": office1,
        "office2": office2,
        "service1": service1,
        "service2": service2,
        "now": now
    }

def test_overall_analytics(setup_data):
    resp = client.get("/api/v1/analytics/overall")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tokens"] == 6
    assert data["waiting_tokens"] == 1
    assert data["called_tokens"] == 1
    assert data["serving_tokens"] == 1
    assert data["completed_tokens"] == 1
    assert data["cancelled_tokens"] == 1
    assert data["skipped_tokens"] == 1
    
    assert data["average_wait_minutes"] == 15.0 # (15 + 15) / 2 = 15
    assert data["average_service_minutes"] == 15.0 # (15 - 0) / 1 = 15
    
    assert data["completion_rate"] == (1/6) * 100
    assert data["cancellation_rate"] == (1/6) * 100
    assert data["skip_rate"] == (1/6) * 100

def test_office_analytics(setup_data):
    office1 = setup_data["office1"]
    resp = client.get(f"/api/v1/analytics/offices/{office1.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tokens"] == 4
    assert data["current_waiting_count"] == 1
    assert data["current_serving_count"] == 1
    assert data["completed_tokens"] == 1
    assert data["average_wait_minutes"] == 15.0

def test_service_analytics(setup_data):
    service2 = setup_data["service2"]
    resp = client.get(f"/api/v1/analytics/services/{service2.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tokens"] == 2
    assert data["cancelled_tokens"] == 1
    assert data["skipped_tokens"] == 1

def test_queue_analytics(setup_data):
    office1 = setup_data["office1"]
    service1 = setup_data["service1"]
    resp = client.get(f"/api/v1/analytics/queues/{office1.id}/{service1.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["waiting_count"] == 1
    assert data["called_count"] == 1
    assert data["serving_count"] == 1
    assert data["active_counters"] == 3

def test_ml_monitoring(setup_data):
    resp = client.get("/api/v1/analytics/ml-monitoring")
    assert resp.status_code == 200
    data = resp.json()
    # Wait, tokens 5 and 6 didn't have predicted_wait_minutes specified in setup, they should be None.
    # Ah, I specified predicted_wait_minutes for 1,2,3,4.
    assert data["total_with_predictions"] == 4
    assert data["total_without_predictions"] == 2
    
    # MAE check: token 4 completed. Actual wait = 15 mins. Predicted = 20 mins. Error = 5.
    # So MAE = 5. RMSE = 5.
    assert data["mae_minutes"] == 5.0
    assert data["rmse_minutes"] == 5.0

def test_date_filtering(setup_data):
    now = setup_data["now"]
    from urllib.parse import quote
    start = quote((now - timedelta(minutes=5)).isoformat())
    resp = client.get(f"/api/v1/analytics/overall?start_date={start}")
    assert resp.status_code == 200
    data = resp.json()
    # Tokens 1, 5, 6 were created at `now`. Tokens 2,3,4 were created earlier.
    assert data["total_tokens"] == 3

def test_invalid_date_filtering(setup_data):
    now = setup_data["now"]
    from urllib.parse import quote
    start = quote(now.isoformat())
    end = quote((now - timedelta(days=1)).isoformat())
    resp = client.get(f"/api/v1/analytics/overall?start_date={start}&end_date={end}")
    assert resp.status_code == 400

def test_invalid_office():
    resp = client.get("/api/v1/analytics/offices/9999")
    assert resp.status_code == 404

def test_invalid_service():
    resp = client.get("/api/v1/analytics/services/9999")
    assert resp.status_code == 404
