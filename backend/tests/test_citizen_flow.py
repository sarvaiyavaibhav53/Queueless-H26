import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter

client = TestClient(app, raise_server_exceptions=True)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    db.query(QueueToken).delete()
    db.query(CounterService).delete()
    db.query(Counter).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    yield db
    db.query(QueueToken).delete()
    db.query(CounterService).delete()
    db.query(Counter).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    db.close()

@pytest.fixture(scope="module")
def setup_data(db_session: Session):
    office = Office(code="OC", name="Citizen Office", counters=3, demand_multiplier=1.0)
    service = Service(code="SC", name="Citizen Service", mean_minutes=10.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    counter = QueueCounter(office_id=office.id, service_id=service.id, current_number=0)
    db_session.add(counter)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_office_discovery(setup_data):
    resp = client.get("/api/v1/offices")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    codes = [o["code"] for o in data]
    assert "OC" in codes

def test_service_discovery(setup_data):
    resp = client.get("/api/v1/services")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    codes = [s["code"] for s in data]
    assert "SC" in codes

def test_queue_status(setup_data):
    resp = client.get(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["waiting_count"] == 0
    assert data["active_counters"] == 3

    # Join queue to see if waiting_count increases
    resp_join = client.post(
        f"/api/v1/queues/{setup_data['service'].id}/join",
        json={"office_id": setup_data['office'].id}
    )
    print("JOIN RESPONSE:", resp_join.status_code, resp_join.text)

    resp = client.get(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["waiting_count"] == 1
    
    # Invalid IDs
    resp = client.get("/api/v1/queues/999/999/status")
    assert resp.status_code == 404

