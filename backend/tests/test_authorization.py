import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import SessionLocal
from app.models.user import User, UserRole
from app.models.office import Office
from app.models.service import Service
from app.core.security import get_password_hash

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_auth_data():
    db = SessionLocal()
    
    import uuid
    uid = str(uuid.uuid4())[:8]
    cit_a = User(email=f"cita_{uid}@example.com", password_hash=get_password_hash("pass"), role=UserRole.citizen)
    cit_b = User(email=f"citb_{uid}@example.com", password_hash=get_password_hash("pass"), role=UserRole.citizen)
    emp = User(email=f"emp_{uid}@example.com", password_hash=get_password_hash("pass"), role=UserRole.employee)
    
    db.add_all([cit_a, cit_b, emp])
    db.commit()
    
    o = Office(code="AUTH", name="Auth Office", counters=1)
    s = Service(code="AUTHS", name="Auth Service", mean_minutes=15)
    db.add(o)
    db.add(s)
    db.commit()
    
    yield {
        "cit_a": cit_a.email,
        "cit_b": cit_b.email,
        "emp": emp.email,
        "office_id": o.id,
        "service_id": s.id
    }
    
    db.close()

def get_token(email):
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "pass"})
    return resp.json()["access_token"]

def test_authorization_flows(setup_auth_data):
    app.dependency_overrides.clear()
    
    cit_a_token = get_token(setup_auth_data["cit_a"])
    cit_b_token = get_token(setup_auth_data["cit_b"])
    emp_token = get_token(setup_auth_data["emp"])
    
    office_id = setup_auth_data["office_id"]
    service_id = setup_auth_data["service_id"]
    
    # Citizen A joins
    headers_a = {"Authorization": f"Bearer {cit_a_token}"}
    join_resp = client.post(f"/api/v1/queues/{service_id}/join", json={"office_id": office_id}, headers=headers_a)
    assert join_resp.status_code == 201
    token_id = join_resp.json()["id"]
    
    # Citizen A views own token
    view_resp = client.get(f"/api/v1/tokens/{token_id}", headers=headers_a)
    assert view_resp.status_code == 200
    
    # Citizen B views A's token -> 403
    headers_b = {"Authorization": f"Bearer {cit_b_token}"}
    view_resp_b = client.get(f"/api/v1/tokens/{token_id}", headers=headers_b)
    assert view_resp_b.status_code == 403
    
    # Citizen A calls next -> 403 (employee only)
    call_resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next", headers=headers_a)
    assert call_resp.status_code == 403
    
    # Citizen A views analytics -> 403 (admin only)
    analytics_resp = client.get("/api/v1/analytics/overall", headers=headers_a)
    assert analytics_resp.status_code == 403
    
    # Employee calls next
    headers_emp = {"Authorization": f"Bearer {emp_token}"}
    call_emp_resp = client.post(f"/api/v1/queues/{office_id}/{service_id}/call-next", headers=headers_emp)
    assert call_emp_resp.status_code == 200
    
    # Employee views analytics -> 403 (admin only)
    analytics_emp_resp = client.get("/api/v1/analytics/overall", headers=headers_emp)
    assert analytics_emp_resp.status_code == 403
    
    # WebSocket auth tests
    # No token -> close
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/queues/{office_id}/{service_id}"):
            pass
            
    # Valid token -> connect
    with client.websocket_connect(f"/ws/queues/{office_id}/{service_id}?token={cit_a_token}") as ws:
        # should connect
        pass
