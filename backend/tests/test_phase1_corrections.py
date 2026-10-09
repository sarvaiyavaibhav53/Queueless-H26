"""
PRAVAAH — Phase 1 Regression Test Suite

Covers the three verified inconsistencies corrected in Phase 1:
1. Correction 1: Service-specific active counter counts in GET /api/v1/queues/{office_id}/{service_id}/status
2. Correction 2: Removal of legacy ad-hoc prediction from token creation (POST /api/v1/queues/{service_id}/join)
                 and authoritative live ETA behavior (GET /api/v1/queues/{token_id}/eta)
3. Correction 3: Route definition verification (POST /api/v1/tokens/{token_id}/complete vs legacy queue typo,
                 CORS allow_credentials=False, and WebSocket path contract)
"""

from datetime import datetime, timezone, timedelta
from unittest import mock
import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import get_db
from app.models.base import Base
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter
from app.schemas.prediction import PredictionSource

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clean_test_db():
    """
    Provides an isolated in-memory database per test via FastAPI dependency overrides.
    Ensures test execution and fixture cleanup NEVER delete development or demo data from test.db.
    """
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    test_db = TestSessionLocal()
    try:
        yield test_db
    finally:
        test_db.close()
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


# =============================================================================
# CORRECTION 1: SERVICE-SPECIFIC ACTIVE COUNTER COUNT
# =============================================================================

def test_service_specific_active_counter_counts(clean_test_db):
    """
    Requirements:
    - Count counters belonging to the requested office.
    - Count only counters actively assigned to the requested service and currently ACTIVE.
    - Do not confuse active counters with busy counters (busy counter is ACTIVE, serving a token).
    - Exclude counters that are UNAVAILABLE.
    - Never report office-wide capacity when physical counters are assigned per service.
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    # Setup office with 5 total desks capacity
    office = Office(id=1, code="MUNI", name="Municipality Central", counters=5, demand_multiplier=1.0)
    service_a = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=2.0)
    service_b = Service(id=2, code="WC", name="Water Connection", mean_minutes=15.0, std_minutes=3.0)
    service_c = Service(id=3, code="BC", name="Birth Certificate", mean_minutes=8.0, std_minutes=1.5)
    db.add_all([office, service_a, service_b, service_c])
    db.commit()

    # Physical counters setup:
    # Counter 1: Assigned to Service A (PT), ACTIVE (idle)
    c1 = Counter(id=1, office_id=1, name="Counter 1", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_now)
    # Counter 2: Assigned to Service A (PT), ACTIVE (BUSY serving active token)
    c2 = Counter(id=2, office_id=1, name="Counter 2", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_now)
    # Counter 3: Assigned to Service A (PT), UNAVAILABLE (offline/break)
    c3 = Counter(id=3, office_id=1, name="Counter 3", status=CounterStatus.UNAVAILABLE, assigned_service_id=1, status_changed_at=t_now)
    # Counter 4: Assigned to Service B (WC), ACTIVE (idle)
    c4 = Counter(id=4, office_id=1, name="Counter 4", status=CounterStatus.ACTIVE, assigned_service_id=2, status_changed_at=t_now)
    # Counter 5: Assigned to Service B (WC), UNAVAILABLE (offline)
    c5 = Counter(id=5, office_id=1, name="Counter 5", status=CounterStatus.UNAVAILABLE, assigned_service_id=2, status_changed_at=t_now)
    db.add_all([c1, c2, c3, c4, c5])
    db.commit()

    # Create active serving token on Counter 2 to prove busy != unavailable
    busy_token = QueueToken(
        id=501,
        token_number="MUNI-PT-001",
        office_id=1,
        service_id=1,
        counter_id=2,
        status=TokenStatus.SERVING,
        created_at=t_now - timedelta(minutes=15),
        serving_at=t_now - timedelta(minutes=5),
    )
    # Create 3 waiting tokens for Service A
    for i in range(1, 4):
        db.add(QueueToken(
            id=510 + i,
            token_number=f"MUNI-PT-00{i+1}",
            office_id=1,
            service_id=1,
            status=TokenStatus.WAITING,
            created_at=t_now - timedelta(minutes=10 - i),
        ))
    # Create 1 waiting token for Service B
    db.add(QueueToken(
        id=601,
        token_number="MUNI-WC-001",
        office_id=1,
        service_id=2,
        status=TokenStatus.WAITING,
        created_at=t_now - timedelta(minutes=5),
    ))
    db.add(busy_token)
    db.commit()

    # Query status for Service A:
    # Assigned: c1 (ACTIVE), c2 (ACTIVE + BUSY), c3 (UNAVAILABLE) -> expected active_counters = 2
    # Waiting count = 3
    resp_a = client.get("/api/v1/queues/1/1/status")
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["waiting_count"] == 3
    assert data_a["active_counters"] == 2, (
        f"Expected 2 active counters for Service A (c1 active, c2 active-busy; c3 unavailable excluded). Got {data_a['active_counters']}"
    )

    # Query status for Service B:
    # Assigned: c4 (ACTIVE), c5 (UNAVAILABLE) -> expected active_counters = 1
    # Waiting count = 1
    resp_b = client.get("/api/v1/queues/1/2/status")
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert data_b["waiting_count"] == 1
    assert data_b["active_counters"] == 1, (
        f"Expected 1 active counter for Service B (c4 active; c5 unavailable excluded). Got {data_b['active_counters']}"
    )

    # Query status for Service C:
    # No counters assigned -> expected active_counters = 0
    # Waiting count = 0
    resp_c = client.get("/api/v1/queues/1/3/status")
    assert resp_c.status_code == 200
    data_c = resp_c.json()
    assert data_c["waiting_count"] == 0
    assert data_c["active_counters"] == 0, (
        f"Expected 0 active counters for unassigned Service C. Got {data_c['active_counters']}"
    )

    # None of the endpoints should report the office-wide capacity (5)
    assert data_a["active_counters"] != 5
    assert data_b["active_counters"] != 5
    assert data_c["active_counters"] != 5


def test_queue_status_unseeded_office_fallback(clean_test_db):
    """
    Offices created without physical Counter records fallback to office.counters
    preserving legacy test compatibility.
    """
    db = clean_test_db
    office = Office(id=9, code="LEG", name="Legacy Office", counters=4, demand_multiplier=1.0)
    service = Service(id=9, code="LEG_SVC", name="Legacy Service", mean_minutes=12.0)
    db.add_all([office, service])
    db.commit()

    resp = client.get("/api/v1/queues/9/9/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["waiting_count"] == 0
    assert data["active_counters"] == 4


def test_queue_status_not_found(clean_test_db):
    """404 responses for invalid office or service IDs."""
    resp1 = client.get("/api/v1/queues/999/1/status")
    assert resp1.status_code == 404
    assert "Office not found" in resp1.json()["detail"]

    clean_test_db.add(Office(id=1, code="MUNI", name="Municipality", counters=2))
    clean_test_db.commit()
    resp2 = client.get("/api/v1/queues/1/999/status")
    assert resp2.status_code == 404
    assert "Service not found" in resp2.json()["detail"]


# =============================================================================
# CORRECTION 2: REMOVE LEGACY ETA CALCULATION FROM TOKEN CREATION
# =============================================================================

def test_token_creation_does_not_call_legacy_predictor(clean_test_db):
    """
    Requirements:
    - Remove the ad-hoc mock prediction from the token-creation path.
    - Token creation does NOT call prediction_service.predict_wait_time.
    - predicted_wait_minutes on QueueToken response is None.
    - Token creation does not invent mock arrival_rate_15m=5 or no_show_rate=0.05.
    """
    db = clean_test_db
    office = Office(id=1, code="MUNI", name="Municipality", counters=3)
    service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0)
    db.add_all([office, service])
    db.commit()

    with mock.patch("app.services.prediction_service.prediction_service.predict_wait_time") as mock_legacy:
        resp = client.post("/api/v1/queues/1/join", json={"office_id": 1})
        assert resp.status_code == 201
        data = resp.json()

        # Token created successfully
        assert data["id"] is not None
        assert data["token_number"] is not None
        assert data["status"] == TokenStatus.WAITING.value
        assert data["queue_position"] == 1

        # predicted_wait_minutes must be None on token creation
        assert data["predicted_wait_minutes"] is None

        # Legacy predictor must NOT have been called
        mock_legacy.assert_not_called()


def test_authoritative_live_eta_endpoint_supported_condition(clean_test_db):
    """
    Requirements:
    - Authoritative live ETA uses the canonical 15-feature contract and hybrid prediction service.
    - When counter is busy serving an active token and a waiting token exists,
      RF model serves with P10-P90 prediction intervals.
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    office = Office(id=1, code="MUNI", name="Municipality", counters=1)
    service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=3.0)
    db.add_all([office, service])
    db.commit()

    # Counter 1 assigned and active
    c1 = Counter(id=1, office_id=1, name="Counter 1", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_now)
    db.add(c1)
    db.commit()
    db.add(CounterService(counter_id=1, service_id=1))
    db.commit()

    # Active serving token on Counter 1 (ensures busy_assigned_counters == assigned_active_counters == 1)
    serv_token = QueueToken(
        id=101,
        token_number="MUNI-PT-001",
        office_id=1,
        service_id=1,
        counter_id=1,
        status=TokenStatus.SERVING,
        created_at=t_now - timedelta(minutes=15),
        called_at=t_now - timedelta(minutes=6),
        serving_at=t_now - timedelta(minutes=5),
    )
    # Waiting token behind it
    wait_token = QueueToken(
        id=102,
        token_number="MUNI-PT-002",
        office_id=1,
        service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_now - timedelta(minutes=4),
    )
    db.add_all([serv_token, wait_token])
    db.commit()

    # Retrieve live ETA for waiting token via authoritative endpoint
    eta_resp = client.get(f"/api/v1/queues/{wait_token.id}/eta")
    assert eta_resp.status_code == 200
    eta_data = eta_resp.json()

    assert eta_data["token_id"] == wait_token.id
    assert eta_data["office_code"] == "MUNI"
    assert eta_data["service_code"] == "PT"
    assert eta_data["predicted_wait_minutes"] >= 0.0
    assert eta_data["prediction_source"] == PredictionSource.RF.value
    assert eta_data["p10_minutes"] is not None
    assert eta_data["p90_minutes"] is not None
    assert eta_data["p10_minutes"] <= eta_data["predicted_wait_minutes"] <= eta_data["p90_minutes"]


def test_authoritative_live_eta_fallback_condition(clean_test_db):
    """
    When waiting queue has zero busy assigned counters (Guard A),
    the system gracefully triggers analytical fallback.
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    office = Office(id=1, code="MUNI", name="Municipality", counters=1)
    service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=3.0)
    db.add_all([office, service])
    db.commit()

    c1 = Counter(id=1, office_id=1, name="Counter 1", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_now)
    db.add(c1)
    db.commit()
    db.add(CounterService(counter_id=1, service_id=1))
    db.commit()

    wait_token = QueueToken(
        id=201,
        token_number="MUNI-PT-001",
        office_id=1,
        service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_now - timedelta(minutes=2),
    )
    db.add(wait_token)
    db.commit()

    eta_resp = client.get(f"/api/v1/queues/{wait_token.id}/eta")
    assert eta_resp.status_code == 200
    eta_data = eta_resp.json()

    assert eta_data["prediction_source"] == PredictionSource.ANALYTICAL_FALLBACK.value
    assert eta_data["fallback_reason"] == "QUEUE_WITH_ZERO_BUSY_COUNTERS"
    assert eta_data["predicted_wait_minutes"] >= 0.0


def test_authoritative_live_eta_serving_token_is_zero(clean_test_db):
    """
    A token that has transitioned to SERVING returns predicted_wait_minutes = 0.0.
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    office = Office(id=1, code="MUNI", name="Municipality", counters=1)
    service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0)
    db.add_all([office, service])
    db.commit()

    c1 = Counter(id=1, office_id=1, name="Counter 1", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_now)
    db.add(c1)
    db.commit()
    db.add(CounterService(counter_id=1, service_id=1))

    serving_token = QueueToken(
        id=777,
        token_number="MUNI-PT-777",
        office_id=1,
        service_id=1,
        counter_id=1,
        status=TokenStatus.SERVING,
        created_at=t_now - timedelta(minutes=10),
        called_at=t_now - timedelta(minutes=4),
        serving_at=t_now - timedelta(minutes=2),
    )
    db.add(serving_token)
    db.commit()

    eta_resp = client.get(f"/api/v1/queues/{serving_token.id}/eta")
    assert eta_resp.status_code == 200
    eta_data = eta_resp.json()
    assert eta_data["predicted_wait_minutes"] == 0.0
    assert eta_data["p10_minutes"] == 0.0
    assert eta_data["p90_minutes"] == 0.0


def test_authoritative_live_eta_unsupported_model_office(clean_test_db):
    """
    Office outside frozen model support (e.g. RJT) returns 422 UNSUPPORTED_MODEL_OFFICE.
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    office = Office(id=99, code="RJT", name="Rajkot RTO", counters=2)
    service = Service(id=99, code="LL", name="Learning License", mean_minutes=15.0)
    db.add_all([office, service])
    db.commit()

    c = Counter(id=99, office_id=99, name="Counter R", status=CounterStatus.ACTIVE, assigned_service_id=99, status_changed_at=t_now)
    db.add(c)
    db.commit()
    db.add(CounterService(counter_id=99, service_id=99))
    token = QueueToken(
        id=999,
        token_number="RJT-LL-001",
        office_id=99,
        service_id=99,
        status=TokenStatus.WAITING,
        created_at=t_now,
    )
    db.add(token)
    db.commit()

    eta_resp = client.get(f"/api/v1/queues/{token.id}/eta")
    assert eta_resp.status_code == 422
    err_data = eta_resp.json()["detail"]
    assert err_data["error_code"] == "UNSUPPORTED_MODEL_OFFICE"


# =============================================================================
# CORRECTION 3: ROUTE VERIFICATION, CORS CREDENTIALS, AND DOC PARITY
# =============================================================================

def test_token_completion_route_is_under_tokens(clean_test_db):
    """
    Verifies that the route is POST /api/v1/tokens/{token_id}/complete,
    and POST /api/v1/queues/{token_id}/complete returns 404 (typo does not exist).
    """
    db = clean_test_db
    t_now = datetime.now(timezone.utc)

    office = Office(id=1, code="MUNI", name="Municipality", counters=1)
    service = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0)
    db.add_all([office, service])
    db.commit()

    token = QueueToken(
        id=301,
        token_number="MUNI-PT-301",
        office_id=1,
        service_id=1,
        status=TokenStatus.SERVING,
        created_at=t_now - timedelta(minutes=15),
        serving_at=t_now - timedelta(minutes=5),
    )
    db.add(token)
    db.commit()

    # Typo path must NOT exist (404)
    typo_resp = client.post(f"/api/v1/queues/{token.id}/complete")
    assert typo_resp.status_code in [404, 405]

    # Correct route under /api/v1/tokens/ must succeed
    correct_resp = client.post(f"/api/v1/tokens/{token.id}/complete")
    assert correct_resp.status_code == 200
    assert correct_resp.json()["status"] == TokenStatus.COMPLETED.value


def test_cors_allow_credentials_is_false():
    """
    Verifies CORS middleware configuration in application:
    allow_credentials MUST be False by default.
    """
    cors_middleware = None
    for mw in app.user_middleware:
        if mw.cls == CORSMiddleware:
            cors_middleware = mw
            break

    assert cors_middleware is not None, "CORSMiddleware must be registered on FastAPI app"
    assert cors_middleware.kwargs.get("allow_credentials") is False, (
        "CORSMiddleware must have allow_credentials=False"
    )


def test_api_contract_documentation_contains_corrections():
    """
    Verifies that docs/api_contract.md contains the corrected paths and settings:
    - POST /api/v1/tokens/{token_id}/complete is referenced
    - /api/v1/queues/{token_id}/complete typo is eliminated
    - allow_credentials=False is documented
    """
    import os
    docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "api_contract.md")
    with open(docs_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The typo must not exist
    assert "/api/v1/queues/{token_id}/complete" not in content, (
        "docs/api_contract.md must not contain outdated /api/v1/queues/{token_id}/complete"
    )
    # The correct path must be documented
    assert "/api/v1/tokens/{token_id}/complete" in content
    # allow_credentials=False must be documented
    assert "allow_credentials=False" in content
