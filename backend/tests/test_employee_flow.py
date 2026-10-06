import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter

client = TestClient(app, raise_server_exceptions=True)

test_state = {}

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
    office = Office(code="EMP", name="Employee Office", counters=3, demand_multiplier=1.0)
    service = Service(code="SRV", name="Employee Service", mean_minutes=10.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_employee_queue_view_empty(setup_data):
    resp = client.get(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/tokens")
    assert resp.status_code == 200
    assert len(resp.json()) == 0

def test_create_tokens(setup_data):
    # Join 3 tokens
    for i in range(3):
        resp = client.post(
            f"/api/v1/queues/{setup_data['service'].id}/join",
            json={"office_id": setup_data['office'].id}
        )
        assert resp.status_code == 201

def test_employee_queue_view(setup_data):
    resp = client.get(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/tokens")
    assert resp.status_code == 200
    tokens = resp.json()
    assert len(tokens) == 3
    for t in tokens:
        assert t["status"] == "WAITING"
        assert t["queue_position"] > 0
    test_state["token1_id"] = tokens[0]["id"]
    test_state["token2_id"] = tokens[1]["id"]
    test_state["token3_id"] = tokens[2]["id"]

def test_call_next_ordering(setup_data):
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == test_state["token1_id"]
    assert data["status"] == "CALLED"
    
    # Second token should now be position 1
    pos_resp = client.get(f"/api/v1/tokens/{test_state['token2_id']}/position")
    assert pos_resp.json()["position"] == 1

def test_start_service(setup_data):
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "SERVING"

def test_complete_service(setup_data):
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/complete")
    assert resp.status_code == 200
    assert resp.json()["status"] == "COMPLETED"

def test_employee_queue_view_after_complete(setup_data):
    resp = client.get(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/tokens")
    assert resp.status_code == 200
    tokens = resp.json()
    # token1 is COMPLETED, so it should not be in the active list
    assert len(tokens) == 2
    assert tokens[0]["id"] == test_state["token2_id"]

def test_skip_token(setup_data):
    # Call next (should be token 2)
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.json()["id"] == test_state["token2_id"]
    
    # Skip it
    skip_resp = client.post(f"/api/v1/tokens/{test_state['token2_id']}/skip")
    assert skip_resp.status_code == 200
    assert skip_resp.json()["status"] == "SKIPPED"
    
    # Check if token 3 advanced to position 1
    pos_resp = client.get(f"/api/v1/tokens/{test_state['token3_id']}/position")
    assert pos_resp.json()["position"] == 1

def test_invalid_transitions(setup_data):
    # Token 1 is COMPLETED
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/start")
    assert resp.status_code == 400
    assert "Invalid state transition" in resp.json()["detail"]
    
    # Token 2 is SKIPPED
    resp = client.post(f"/api/v1/tokens/{test_state['token2_id']}/complete")
    assert resp.status_code == 400
    assert "Invalid state transition" in resp.json()["detail"]
    
    # Token 3 is WAITING
    resp = client.post(f"/api/v1/tokens/{test_state['token3_id']}/start")
    assert resp.status_code == 400
    assert "Invalid state transition" in resp.json()["detail"]
    
    # Call token 3
    client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    
    # Now token 3 is CALLED. We cannot complete it without starting.
    resp = client.post(f"/api/v1/tokens/{test_state['token3_id']}/complete")
    assert resp.status_code == 400
    assert "Invalid state transition" in resp.json()["detail"]

def test_empty_queue(setup_data):
    # Cancel token 3 so it's not active
    # Wait, it's CALLED. We can't cancel a CALLED token. We can skip or complete.
    client.post(f"/api/v1/tokens/{test_state['token3_id']}/skip")
    
    # Now queue is empty
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No waiting tokens found in this queue"
