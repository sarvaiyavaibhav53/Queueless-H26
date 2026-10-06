import pytest
import datetime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
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
    
    # Create test offices and services for isolation testing
    rjt = Office(code="RJT", name="Rajkot Office", counters=3, demand_multiplier=1.0)
    amd = Office(code="AMD", name="Ahmedabad Office", counters=2, demand_multiplier=1.0)
    ll = Service(code="LL", name="Learner License", mean_minutes=10.0)
    veh = Service(code="VEH", name="Vehicle Registration", mean_minutes=15.0)
    
    db.add_all([rjt, amd, ll, veh])
    db.commit()
    db.refresh(rjt)
    db.refresh(amd)
    db.refresh(ll)
    db.refresh(veh)
    
    yield {
        "db": db,
        "rjt": rjt,
        "amd": amd,
        "ll": ll,
        "veh": veh
    }
    
    db.query(QueueToken).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    db.close()

def test_e2e_scenario_a_successful_journey(db_session):
    """
    discover offices -> discover services -> join queue -> receive token -> verify wait
    employee call-next -> start -> complete -> verify analytics
    """
    office_id = db_session["rjt"].id
    service_id = db_session["ll"].id

    # Join queue
    join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
    assert join_resp.status_code == 201
    token = join_resp.json()
    assert token["token_number"].startswith("RJT-LL-")
    assert token["status"] == "WAITING"
    assert isinstance(token["predicted_wait_minutes"], (int, float, type(None)))
    
    token_id = token["id"]
    
    # Position
    pos_resp = client.get(f"/api/v1/tokens/{token_id}/position")
    assert pos_resp.json()["position"] == 1
    
    # Call-next
    call_resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next")
    assert call_resp.status_code == 200
    called_token = call_resp.json()
    assert called_token["id"] == token_id
    assert called_token["status"] == "CALLED"
    
    # Start
    start_resp = client.post(f"/api/v1/tokens/{token_id}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "SERVING"
    
    # Complete
    complete_resp = client.post(f"/api/v1/tokens/{token_id}/complete")
    assert complete_resp.status_code == 200
    assert complete_resp.json()["status"] == "COMPLETED"
    
    # Verify Analytics
    analytics_resp = client.get("/api/v1/analytics/overall").json()
    assert analytics_resp["completed_tokens"] >= 1
    assert analytics_resp["average_wait_minutes"] is not None

def test_e2e_scenario_b_cancellation(db_session):
    """join -> cancel -> verify analytics"""
    office_id = db_session["rjt"].id
    service_id = db_session["ll"].id

    # Join
    join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
    token_id = join_resp.json()["id"]
    
    # Cancel
    cancel_resp = client.post(f"/api/v1/tokens/{token_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"
    
    # Analytics
    analytics_resp = client.get("/api/v1/analytics/overall").json()
    assert analytics_resp["cancelled_tokens"] >= 1

def test_e2e_scenario_c_skip(db_session):
    """join -> call-next -> skip"""
    office_id = db_session["rjt"].id
    service_id = db_session["ll"].id

    while client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").status_code == 200:
        pass

    join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
    token_id = join_resp.json()["id"]
    
    call_resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next")
    assert call_resp.json()["id"] == token_id
    
    skip_resp = client.post(f"/api/v1/tokens/{token_id}/skip")
    assert skip_resp.status_code == 200
    assert skip_resp.json()["status"] == "SKIPPED"

def test_e2e_scenario_d_multiple_citizens(db_session):
    office_id = db_session["rjt"].id
    service_id = db_session["ll"].id
    
    # Ensure queue is empty first
    while client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").status_code == 200:
        pass
        
    t1 = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id}).json()
    t2 = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id}).json()
    t3 = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id}).json()
    
    assert t1["queue_position"] == 1
    assert t2["queue_position"] == 2
    assert t3["queue_position"] == 3
    
    # Cancel t2
    client.post(f"/api/v1/tokens/{t2['id']}/cancel")
    
    # t3 position should now be 2
    pos3 = client.get(f"/api/v1/tokens/{t3['id']}/position").json()
    assert pos3["position"] == 2
    
    # Call next should give t1
    c1 = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").json()
    assert c1["id"] == t1["id"]
    
    # Call next should give t3
    c3 = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").json()
    assert c3["id"] == t3["id"]

def test_e2e_scenario_e_isolation(db_session):
    rjt_id = db_session["rjt"].id
    amd_id = db_session["amd"].id
    ll_id = db_session["ll"].id
    veh_id = db_session["veh"].id
    
    # RJT + LL
    tr = client.post(f"/api/v1/queues/{ll_id}/join", json={"office_id": rjt_id}).json()
    
    # AMD + VEH
    ta = client.post(f"/api/v1/queues/{veh_id}/join", json={"office_id": amd_id}).json()
    
    # Call next in AMD+VEH
    ca = client.post(f"/api/v1/queues/{amd_id}/{veh_id}/call-next").json()
    assert ca["id"] == ta["id"]
    
    # RJT+LL should still be waiting
    pos_r = client.get(f"/api/v1/tokens/{tr['id']}/position").json()
    assert pos_r["position"] >= 1

def test_websocket_e2e(db_session):
    office_id = db_session["rjt"].id
    service_id = db_session["ll"].id
    
    # Ensure queue is empty first
    while client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next").status_code == 200:
        pass
        
    with client.websocket_connect(f"/ws/queues/{office_id}/{service_id}") as websocket:
        # Join
        join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id})
        token_id = join_resp.json()["id"]
        
        ws_msg = websocket.receive_json()
        assert ws_msg["event"] == "JOINED"
        assert any(t["token_id"] == token_id for t in ws_msg["tokens"])
        
        # Call
        client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next")
        ws_msg = websocket.receive_json()
        assert ws_msg["event"] == "CALLED"
        assert any(t["token_id"] == token_id for t in ws_msg["tokens"])
        
        # Start
        client.post(f"/api/v1/tokens/{token_id}/start")
        ws_msg = websocket.receive_json()
        assert ws_msg["event"] == "STARTED"
        assert any(t["token_id"] == token_id for t in ws_msg["tokens"])
        
        # Complete
        client.post(f"/api/v1/tokens/{token_id}/complete")
        ws_msg = websocket.receive_json()
        assert ws_msg["event"] == "COMPLETED"
        assert any(t["token_id"] == token_id for t in ws_msg["tokens"])

def test_input_validation():
    # Nonexistent office
    resp = client.post("/api/v1/queues/1/join", json={"office_id": 9999})
    assert resp.status_code == 400
    
    # Nonexistent token
    resp = client.post("/api/v1/tokens/9999/start")
    assert resp.status_code == 400
    
    # String for ID
    resp = client.get("/api/v1/tokens/invalid_string")
    assert resp.status_code == 422 # Pydantic Validation error
    
    # SQL Injection check
    resp = client.get("/api/v1/tokens/1%20OR%201=1")
    assert resp.status_code == 422
