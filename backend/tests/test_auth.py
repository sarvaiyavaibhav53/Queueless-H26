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


def test_jwt_production_security_validation():
    from app.core.config import Settings

    # Insecure default in production must fail safely
    prod_bad = Settings(ENVIRONMENT="production", JWT_SECRET="super-secret-key-for-development")
    with pytest.raises(ValueError, match="Insecure JWT_SECRET"):
        prod_bad.validate_security()

    # Empty secret in production must fail safely
    prod_empty = Settings(ENVIRONMENT="production", JWT_SECRET="")
    with pytest.raises(ValueError, match="Insecure JWT_SECRET"):
        prod_empty.validate_security()

    # Strong custom secret in production must pass
    prod_good = Settings(ENVIRONMENT="production", JWT_SECRET="c0mpl3x-cr1pt0-pr0d-s3cr3t-k3y-987654321")
    prod_good.validate_security()

    # Convenient local development defaults must pass
    dev_default = Settings(ENVIRONMENT="development", JWT_SECRET="super-secret-key-for-development")
    dev_default.validate_security()


def test_cors_credentials_disabled_by_default():
    from fastapi.middleware.cors import CORSMiddleware
    cors_mw = next((m for m in app.user_middleware if m.cls == CORSMiddleware), None)
    assert cors_mw is not None
    assert cors_mw.kwargs.get("allow_credentials") is False

    # Also test via preflight HTTP OPTIONS request
    resp = client.options(
        "/health/db",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert "access-control-allow-credentials" not in resp.headers

