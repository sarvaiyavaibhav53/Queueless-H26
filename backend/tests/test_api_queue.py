import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter

client = TestClient(app)

# Global test state
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
    office = Office(code="O1", name="Office 1", counters=3, demand_multiplier=1.0)
    service = Service(code="S1", name="Service 1", mean_minutes=10.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_join_queue(setup_data):
    # 1. Join queue successfully
    resp = client.post(
        f"/api/v1/queues/{setup_data['service'].id}/join",
        json={"office_id": setup_data['office'].id}
    )
    assert resp.status_code == 201
    data = resp.json()
    
    # 2. Receive token
    assert "id" in data
    assert "token_number" in data
    
    test_state['token1_id'] = data['id']
    
    # 3. Token starts as WAITING
    assert data["status"] == TokenStatus.WAITING.value
    
    # 4. Token number is correct
    assert data["token_number"] == "O1-S1-001"
    
    # 5. Queue position is correct (first in queue)
    assert data["queue_position"] == 1
    
    # Add a second token to test queue position 2
    resp2 = client.post(
        f"/api/v1/queues/{setup_data['service'].id}/join",
        json={"office_id": setup_data['office'].id}
    )
    data2 = resp2.json()
    assert data2["queue_position"] == 2
    test_state['token2_id'] = data2['id']

def test_get_token_and_position(setup_data):
    # Retrieve the first token
    # 6. Get token
    resp = client.get(f"/api/v1/tokens/{test_state['token1_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_number"] == "O1-S1-001"
    
    # 7. Get token position
    pos_resp = client.get(f"/api/v1/tokens/{test_state['token1_id']}/position")
    assert pos_resp.status_code == 200
    assert pos_resp.json()["position"] == 1

def test_call_next_token(setup_data):
    # 10. Call next token
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 200
    data = resp.json()
    
    # 11. Correct token is selected
    assert data["token_number"] == "O1-S1-001"
    
    # 12. Called token has correct status
    assert data["status"] == TokenStatus.CALLED.value
    
    # Position of second token should now be 1
    pos_resp = client.get(f"/api/v1/tokens/{test_state['token2_id']}/position")
    assert pos_resp.json()["position"] == 1

def test_start_serving_and_complete(setup_data):
    # 13. Start serving
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == TokenStatus.SERVING.value
    
    # 14. Complete token
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/complete")
    assert resp.status_code == 200
    assert resp.json()["status"] == TokenStatus.COMPLETED.value

def test_skip_token(setup_data):
    # Call the second token
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 200
    assert resp.json()["token_number"] == "O1-S1-002"
    
    # 15. Skip token when valid
    resp_skip = client.post(f"/api/v1/tokens/{test_state['token2_id']}/skip")
    assert resp_skip.status_code == 200
    assert resp_skip.json()["status"] == TokenStatus.SKIPPED.value

def test_no_waiting_token(setup_data):
    # 16. No waiting token returns correct response
    resp = client.post(f"/api/v1/queues/{setup_data['office'].id}/{setup_data['service'].id}/call-next")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "No waiting tokens found in this queue"

def test_cancel_eligible_token(setup_data):
    # Create token 3
    resp = client.post(
        f"/api/v1/queues/{setup_data['service'].id}/join",
        json={"office_id": setup_data['office'].id}
    )
    token3_id = resp.json()["id"]
    
    # 8. Cancel eligible token
    cancel_resp = client.post(f"/api/v1/tokens/{token3_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == TokenStatus.CANCELLED.value

def test_invalid_operations(setup_data):
    # 17. Invalid token ID
    resp = client.get("/api/v1/tokens/999999")
    assert resp.status_code == 404
    
    # 18. Invalid service ID for joining
    resp = client.post("/api/v1/queues/999999/join", json={"office_id": setup_data['office'].id})
    assert resp.status_code == 400
    
    # 20. Invalid state transition
    # Token 1 is COMPLETED, try to complete again (9. Cannot cancel an invalid/completed token)
    resp = client.post(f"/api/v1/tokens/{test_state['token1_id']}/cancel")
    assert resp.status_code == 400
    assert "Invalid state transition" in resp.json()["detail"]
    
    # Create a WAITING token
    resp = client.post(
        f"/api/v1/queues/{setup_data['service'].id}/join",
        json={"office_id": setup_data['office'].id}
    )
    waiting_token_id = resp.json()["id"]
    
    # 21. Attempt to start WAITING token directly
    resp = client.post(f"/api/v1/tokens/{waiting_token_id}/start")
    assert resp.status_code == 400
    
    # 22. Attempt to complete WAITING token
    resp = client.post(f"/api/v1/tokens/{waiting_token_id}/complete")
    assert resp.status_code == 400

