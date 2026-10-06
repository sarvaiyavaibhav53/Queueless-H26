import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
import asyncio
from unittest.mock import patch

client = TestClient(app)

from app.db.database import SessionLocal
from app.models.queue_counter import QueueCounter

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

@pytest.fixture(autouse=True)
def clear_tokens(db_session: Session):
    db_session.query(QueueToken).delete()
    db_session.commit()

@pytest.fixture(scope="module")
def setup_data(db_session: Session):
    office = Office(code="RJT", name="Rajkot RTO", counters=3, demand_multiplier=1.0)
    service = Service(code="LL", name="Learning License", mean_minutes=15.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return office, service

def test_ws_connection(db_session, setup_data):
    office, service = setup_data
    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as websocket:
        assert websocket

@patch('app.services.prediction_service.WaitTimePredictionService.predict_wait_time', return_value=5.0)
def test_ws_broadcast_on_join(mock_predict, db_session, setup_data):
    office, service = setup_data
    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as websocket:
        response = client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
        assert response.status_code == 201
        
        data = websocket.receive_json()
        assert data["event"] == "JOINED"
        assert data["queue"]["waiting_count"] == 1
        assert len(data["tokens"]) == 1
        assert data["tokens"][0]["status"] == TokenStatus.WAITING.value
        assert data["tokens"][0]["position"] == 1

@patch('app.services.prediction_service.WaitTimePredictionService.predict_wait_time', return_value=5.0)
def test_ws_broadcast_on_call_and_complete(mock_predict, db_session, setup_data):
    office, service = setup_data
    # Create two tokens
    client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
    t2_resp = client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
    t2_id = t2_resp.json()["id"]
    
    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as websocket:
        # Call next
        call_resp = client.post(f"/api/v1/queues/{office.id}/{service.id}/call-next")
        t1_id = call_resp.json()["id"]
        
        data = websocket.receive_json()
        assert data["event"] == "CALLED"
        assert data["queue"]["waiting_count"] == 1
        # Token 2 is waiting (position 1), Token 1 is called (position 0)
        assert len(data["tokens"]) == 2
        
        # Start
        client.post(f"/api/v1/tokens/{t1_id}/start")
        data = websocket.receive_json()
        assert data["event"] == "STARTED"
        
        # Complete
        client.post(f"/api/v1/tokens/{t1_id}/complete")
        data = websocket.receive_json()
        assert data["event"] == "COMPLETED"
        assert data["queue"]["waiting_count"] == 1
        
        # Cancel Token 2
        client.post(f"/api/v1/tokens/{t2_id}/cancel")
        data = websocket.receive_json()
        assert data["event"] == "CANCELLED"
        assert data["queue"]["waiting_count"] == 0
        assert len(data["tokens"]) == 1 # Only the cancelled token is sent
        assert data["tokens"][0]["status"] == "CANCELLED"

def test_ws_isolation(db_session, setup_data):
    office, service = setup_data
    service2 = Service(code="DL", name="Driving License", mean_minutes=15.0)
    db_session.add(service2)
    db_session.commit()
    db_session.refresh(service2)

    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as ws1:
        with client.websocket_connect(f"/ws/queues/{office.id}/{service2.id}") as ws2:
            # Join queue 1
            client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
            
            data1 = ws1.receive_json()
            assert data1["event"] == "JOINED"
            
            # ws2 should not receive this. If we try to receive, it will block. 
            # In a TestClient, if we try to receive and there's nothing, it blocks.
            # But we can verify by triggering something else on ws2.
            client.post(f"/api/v1/queues/{service2.id}/join", json={"office_id": office.id})
            data2 = ws2.receive_json()
            assert data2["event"] == "JOINED"
            assert data2["service_id"] == service2.id

@patch('app.services.prediction_service.WaitTimePredictionService.predict_wait_time', return_value=5.0)
def test_ws_broadcast_on_skip(mock_predict, db_session, setup_data):
    office, service = setup_data
    # Create two tokens
    client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
    t2_resp = client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
    t2_id = t2_resp.json()["id"]
    
    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as websocket:
        # Call next (Token 1)
        call_resp = client.post(f"/api/v1/queues/{office.id}/{service.id}/call-next")
        t1_id = call_resp.json()["id"]
        
        # Read the CALLED message
        websocket.receive_json()
        
        # Skip Token 1
        client.post(f"/api/v1/tokens/{t1_id}/skip")
        data = websocket.receive_json()
        
        assert data["event"] == "SKIPPED"
        assert data["office_id"] == office.id
        assert data["service_id"] == service.id
        assert data["queue"]["waiting_count"] == 1
        
        # We should have Token 2 (WAITING, position 1) and Token 1 (SKIPPED) in the tokens array
        assert len(data["tokens"]) == 2
        
        skipped_token = next(t for t in data["tokens"] if t["token_id"] == t1_id)
        waiting_token = next(t for t in data["tokens"] if t["token_id"] == t2_id)
        
        assert skipped_token["status"] == "SKIPPED"
        assert waiting_token["status"] == "WAITING"
        assert waiting_token["position"] == 1

def test_ws_disconnect_handling(db_session, setup_data):
    office, service = setup_data
    
    from app.ws_manager import ws_manager
    key = f"{office.id}_{service.id}"
    
    # Ensure clean state for this key
    if key in ws_manager.active_connections:
        ws_manager.active_connections[key].clear()
        
    with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as ws_a:
        assert len(ws_manager.active_connections[key]) == 1
        
        with client.websocket_connect(f"/ws/queues/{office.id}/{service.id}") as ws_b:
            assert len(ws_manager.active_connections[key]) == 2
            
        # ws_b context exits, which closes the socket.
        # Trigger an event
        client.post(f"/api/v1/queues/{service.id}/join", json={"office_id": office.id})
        
        # ws_a should receive it
        msg_a = ws_a.receive_json()
        assert msg_a["event"] == "JOINED"
        
        # The connection manager should have cleaned up the dead connection (ws_b)
        assert len(ws_manager.active_connections.get(key, [])) == 1
