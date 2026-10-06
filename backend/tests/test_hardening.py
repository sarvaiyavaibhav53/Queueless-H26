import pytest
import concurrent.futures
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import SessionLocal, engine
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter
from app.services.queue_service import QueueServiceError

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
    office = Office(code="C1", name="Concurrency Office", counters=3, demand_multiplier=1.0)
    service = Service(code="S1", name="Concurrency Service", mean_minutes=10.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_concurrent_join_queue(setup_data, db_session: Session):
    num_requests = 20
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    def join_request(_):
        resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
        return resp.json(), resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(join_request, range(num_requests)))

    # Verification
    successful_joins = [r[0] for r in results if r[1] == 201]
    assert len(successful_joins) == num_requests
    
    token_numbers = [r["token_number"] for r in successful_joins]
    unique_tokens = set(token_numbers)
    
    assert len(unique_tokens) == num_requests, "Duplicate tokens generated!"
    
    # Check DB integrity
    db_count = db_session.query(QueueToken).count()
    assert db_count >= num_requests
    
    # Verify sequence monotonically increased without gaps
    # Token format C1-S1-XXX
    nums = sorted([int(t.split("-")[-1]) for t in token_numbers])
    assert nums == list(range(nums[0], nums[0] + num_requests))

def test_concurrent_call_next(setup_data, db_session: Session):
    # Ensure there are tokens waiting
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # Check how many are waiting
    waiting_count = db_session.query(QueueToken).filter(
        QueueToken.office_id == office_id,
        QueueToken.service_id == service_id,
        QueueToken.status == TokenStatus.WAITING
    ).count()
    
    assert waiting_count > 0, "No waiting tokens for call-next test"
    
    num_calls = min(10, waiting_count)
    
    def call_next_request(_):
        resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next")
        return resp.json(), resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(call_next_request, range(num_calls)))
        
    successful_calls = [r[0] for r in results if r[1] == 200]
    assert len(successful_calls) == num_calls
    
    token_ids = [r["id"] for r in successful_calls]
    unique_ids = set(token_ids)
    
    assert len(unique_ids) == num_calls, "Duplicate tokens assigned to multiple call-next requests!"

def test_transaction_rollback(setup_data, db_session: Session):
    office_id = setup_data["office"].id
    
    # Attempt join with invalid service
    resp = client.post(f"/api/v1/queues/9999/join", json={"office_id": office_id})
    assert resp.status_code == 400
    
    # Verify that counter did not inadvertently increment
    # We do a valid join to see what we get next
    service_id = setup_data["service"].id
    counter_before = db_session.query(QueueCounter).filter_by(office_id=office_id, service_id=service_id).first().current_number
    
    # Do another invalid one just to be sure
    client.post(f"/api/v1/queues/9999/join", json={"office_id": office_id})
    
    counter_after = db_session.query(QueueCounter).filter_by(office_id=office_id, service_id=service_id).first().current_number
    assert counter_before == counter_after, "QueueCounter was incorrectly modified by a failed transaction!"

    # Verify session is still good
    valid_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
    assert valid_resp.status_code == 201

def test_mixed_concurrency(setup_data, db_session: Session):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    def mixed_request(i):
        if i % 2 == 0:
            return client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id}).status_code
        else:
            return client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").status_code
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(mixed_request, range(20)))
        
    assert all(r in [200, 201, 404] for r in results) # 404 if call-next found empty queue

def test_state_transitions(setup_data):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # Join -> WAITING
    join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
    token_id = join_resp.json()["id"]
    
    # Invalid: WAITING -> COMPLETED
    assert client.post(f"/api/v1/tokens/{token_id}/complete").status_code == 400
    
    # Call -> CALLED
    # Since there may be other waiting tokens from previous tests, we just use the API 
    # to specifically force a status change if we were using service directly, 
    # but via API we just call next until we get our token
    while True:
        resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next")
        if resp.status_code == 404 or resp.json()["id"] == token_id:
            break
            
    # Now it is CALLED
    # Invalid: CALLED -> CANCELLED
    assert client.post(f"/api/v1/tokens/{token_id}/cancel").status_code == 400
    
    # CALLED -> SERVING
    assert client.post(f"/api/v1/tokens/{token_id}/start").status_code == 200
    
    # Invalid: SERVING -> SKIPPED
    assert client.post(f"/api/v1/tokens/{token_id}/skip").status_code == 400
    
    # SERVING -> COMPLETED
    assert client.post(f"/api/v1/tokens/{token_id}/complete").status_code == 200
    
    # Invalid: COMPLETED -> CANCELLED
    assert client.post(f"/api/v1/tokens/{token_id}/cancel").status_code == 400

