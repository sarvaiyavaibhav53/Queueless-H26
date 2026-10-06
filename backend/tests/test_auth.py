import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_auth_registration_and_login():
    import uuid
    uid = str(uuid.uuid4())[:8]
    email = f"cit_{uid}@example.com"
    
    # Clear overrides to test real auth
    app.dependency_overrides.clear()
    
    # Register citizen
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "password123",
        "role": "citizen"
    })
    assert reg_resp.status_code == 201
    
    # Duplicate email
    reg_dup = client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "password123",
        "role": "citizen"
    })
    assert reg_dup.status_code == 409
    
    # Try to register admin (should fail)
    reg_admin = client.post("/api/v1/auth/register", json={
        "email": f"hack_{uid}@example.com",
        "password": "password123",
        "role": "admin"
    })
    assert reg_admin.status_code == 403
    
    # Login
    login_resp = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "password123"
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    
    # Get me
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email
