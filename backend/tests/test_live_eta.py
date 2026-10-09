"""
PRAVAAH — Step 2E.4: Live DB -> Snapshot -> Features -> Hybrid Prediction -> ETA Test Suite.

Contains all 15 required focused tests:
- Test 1: Real DB token -> ETA 200
- Test 2: Client state injection cannot override DB state
- Test 3: people_ahead exact training parity (FIFO, target excluded, non-waiting excluded)
- Test 4: queue_length exact active waiting count
- Test 5: Counter state variants (assigned+busy, partially busy / assigned_not_busy_with_waiting_queue, unavailable, zero active)
- Test 6: Elapsed service time (now - service_started_at)
- Test 7: Arrival rates (30m / 60m calculations)
- Test 8: Point-in-time guarantee (future records excluded)
- Test 9: Unsupported application identity -> HTTP 422
- Test 10: Authorization (Citizen A cannot access Citizen B's ETA; employee can)
- Test 11: Deterministic repeated snapshot
- Test 12: Exact 15-feature contract (names, order, count)
- Test 13: Simulator / DB 15-feature exact parity test (Section 23 Hard Requirement)
- Test 14: Office authority (token's stored office wins over conflicting parameter)
- Test 15: Cold-start diagnostic (below-trained-range metadata without policy distortion)
- Test 16: Counter state representation (assigned_not_busy_with_waiting_queue operational semantics)
"""

from datetime import datetime, timedelta, timezone
import math
import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
for p in (BASE_DIR, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from app.db.database import get_db, SessionLocal
from app.models.base import Base
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.user import User, UserRole
from app.core.security import get_password_hash
from app.schemas.prediction import PredictionSource, FallbackReason, IdentityErrorCode
from app.services.prediction_service import (
    WaitTimePredictionService,
    get_prediction_service,
    IdentityValidationError,
)
from app.services.snapshot_adapter import build_simulation_snapshot
from app.services.queue_snapshot_service import (
    QueueSnapshotService,
    get_queue_snapshot_service,
    TokenNotFoundError,
    TokenNotEligibleError,
    PointInTimeViolationError,
)
from shared.features import (
    RAW_FEATURE_NAMES,
    RawFeatureVector,
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
    compute_raw_features,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def clean_db():
    """Provides a clean session per test with rollback/cleanup."""
    db = SessionLocal()
    try:
        # Clean up tables
        db.query(QueueToken).delete()
        db.query(CounterService).delete()
        db.query(Counter).delete()
        db.query(Service).delete()
        db.query(Office).delete()
        db.query(User).delete()
        db.commit()
        yield db
    finally:
        db.rollback()
        db.query(QueueToken).delete()
        db.query(CounterService).delete()
        db.query(Counter).delete()
        db.query(Service).delete()
        db.query(Office).delete()
        db.query(User).delete()
        db.commit()
        db.close()


def setup_standard_muni_env(db, t_snap: datetime):
    """Sets up standard Municipality office with PT, WC, BC services and counters."""
    office = Office(id=1, code="MUNI", name="Municipality Office", counters=3)
    s_pt = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=4.0)
    s_wc = Service(id=2, code="WC", name="Water Service", mean_minutes=15.0, std_minutes=6.0)
    s_bc = Service(id=3, code="BC", name="Birth Certificate", mean_minutes=7.0, std_minutes=2.8)
    db.add_all([office, s_pt, s_wc, s_bc])
    db.commit()

    # Counters
    c1 = Counter(
        id=1,
        office_id=1,
        name="Counter 1",
        status=CounterStatus.ACTIVE,
        assigned_service_id=1,
        status_changed_at=t_snap - timedelta(hours=2),
    )
    c2 = Counter(
        id=2,
        office_id=1,
        name="Counter 2",
        status=CounterStatus.ACTIVE,
        assigned_service_id=1,
        status_changed_at=t_snap - timedelta(hours=2),
    )
    c3 = Counter(
        id=3,
        office_id=1,
        name="Counter 3",
        status=CounterStatus.ACTIVE,
        assigned_service_id=2,
        status_changed_at=t_snap - timedelta(hours=2),
    )
    db.add_all([c1, c2, c3])
    db.commit()

    # Eligibility
    cs1 = CounterService(counter_id=1, service_id=1)
    cs2 = CounterService(counter_id=2, service_id=1)
    cs3 = CounterService(counter_id=3, service_id=2)
    db.add_all([cs1, cs2, cs3])
    db.commit()

    return office, [s_pt, s_wc, s_bc], [c1, c2, c3]


# =============================================================================
# TEST 1: Real DB token -> ETA 200
# =============================================================================
def test_01_real_db_token_eta_200(clean_db, client):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Active counters serving tokens
    t_serv1 = QueueToken(
        id=101, token_number="MUNI-PT-001", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=20),
        called_at=t_snap - timedelta(minutes=6),
        serving_at=t_snap - timedelta(minutes=5),
    )
    t_serv2 = QueueToken(
        id=102, token_number="MUNI-PT-002", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=2,
        created_at=t_snap - timedelta(minutes=15),
        called_at=t_snap - timedelta(minutes=3),
        serving_at=t_snap - timedelta(minutes=2),
    )
    # Waiting token
    t_wait = QueueToken(
        id=103, token_number="MUNI-PT-003", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=10),
    )
    clean_db.add_all([t_serv1, t_serv2, t_wait])

    # Recent arrivals to be in-distribution (10 arrivals in 60m)
    for i in range(10):
        clean_db.add(QueueToken(
            id=200 + i, token_number=f"MUNI-PT-ARR-{i}", office_id=1, service_id=1,
            status=TokenStatus.COMPLETED,
            created_at=t_snap - timedelta(minutes=5 * i + 1),
            serving_at=t_snap - timedelta(minutes=5 * i),
            completed_at=t_snap - timedelta(minutes=5 * i),
        ))
    clean_db.commit()

    service = get_queue_snapshot_service()
    resp = service.get_token_eta(clean_db, token_id=103, now=t_snap)

    assert resp.token_id == 103
    assert resp.office_code == "MUNI"
    assert resp.service_code == "PT"
    assert resp.predicted_wait_minutes > 0.0
    assert resp.prediction_source == PredictionSource.RF.value
    assert resp.fallback_reason is None
    assert resp.people_ahead == 0
    assert resp.queue_length == 1
    assert resp.p10_minutes is not None
    assert resp.p90_minutes is not None
    assert resp.p10_minutes <= resp.predicted_wait_minutes <= resp.p90_minutes

    # Test via API endpoint
    api_res = client.get(f"/api/v1/queues/{t_wait.id}/eta")
    assert api_res.status_code == 200
    data = api_res.json()
    assert data["token_id"] == 103
    assert data["predicted_wait_minutes"] >= 0.0


# =============================================================================
# TEST 2: Client state injection cannot override DB state
# =============================================================================
def test_02_client_cannot_override_prediction_state(clean_db, client):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    t_wait = QueueToken(
        id=105, token_number="MUNI-PT-005", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    # Client attempts to pass spoofed prediction parameters
    spoofed_params = {
        "people_ahead": 999,
        "queue_length": 888,
        "assigned_active_counters": 0,
        "arrival_rate_60m": 500,
        "service_capacity_mu": 1.0,
    }
    resp = client.get(f"/api/v1/queues/{t_wait.id}/eta", params=spoofed_params)
    assert resp.status_code == 200
    data = resp.json()
    # Server DB state wins: people_ahead must be 0, queue_length must be 1
    assert data["people_ahead"] == 0
    assert data["queue_length"] == 1


# =============================================================================
# TEST 3: people_ahead exact training parity
# =============================================================================
def test_03_people_ahead_parity(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # 1 serving token (NOT in waiting queue)
    t_serv = QueueToken(
        id=301, token_number="T-SERV", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=25),
        called_at=t_snap - timedelta(minutes=10),
        serving_at=t_snap - timedelta(minutes=8),
    )
    # 1 completed token
    t_comp = QueueToken(
        id=302, token_number="T-COMP", office_id=1, service_id=1,
        status=TokenStatus.COMPLETED,
        created_at=t_snap - timedelta(minutes=40),
        completed_at=t_snap - timedelta(minutes=20),
    )
    # 1 cancelled token
    t_canc = QueueToken(
        id=303, token_number="T-CANC", office_id=1, service_id=1,
        status=TokenStatus.CANCELLED,
        created_at=t_snap - timedelta(minutes=30),
        cancelled_at=t_snap - timedelta(minutes=15),
    )
    # 3 WAITING tokens in strict FIFO arrival order
    t_w1 = QueueToken(
        id=304, token_number="T-W1", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=15),
    )
    t_w2 = QueueToken(
        id=305, token_number="T-W2", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=10),
    )
    t_w3 = QueueToken(
        id=306, token_number="T-W3", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add_all([t_serv, t_comp, t_canc, t_w1, t_w2, t_w3])
    clean_db.commit()

    service = get_queue_snapshot_service()

    # Exact parity checks:
    # Token 1 (head): 0 people ahead
    eta1 = service.get_token_eta(clean_db, token_id=304, now=t_snap)
    assert eta1.people_ahead == 0
    assert eta1.queue_length == 3

    # Token 2 (mid): 1 person ahead (Token 1)
    eta2 = service.get_token_eta(clean_db, token_id=305, now=t_snap)
    assert eta2.people_ahead == 1
    assert eta2.queue_length == 3

    # Token 3 (tail): 2 people ahead (Tokens 1 and 2)
    eta3 = service.get_token_eta(clean_db, token_id=306, now=t_snap)
    assert eta3.people_ahead == 2
    assert eta3.queue_length == 3


# =============================================================================
# TEST 4: queue_length active count parity
# =============================================================================
def test_04_queue_length_active_count(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    for i in range(5):
        clean_db.add(QueueToken(
            id=400 + i, token_number=f"T-WAIT-{i}", office_id=1, service_id=1,
            status=TokenStatus.WAITING,
            created_at=t_snap - timedelta(minutes=10 - i),
        ))
    # Inactive tokens
    clean_db.add(QueueToken(id=410, token_number="T-SKIPPED", office_id=1, service_id=1, status=TokenStatus.SKIPPED, created_at=t_snap - timedelta(minutes=20)))
    clean_db.add(QueueToken(id=411, token_number="T-COMP", office_id=1, service_id=1, status=TokenStatus.COMPLETED, created_at=t_snap - timedelta(minutes=25)))
    clean_db.commit()

    service = get_queue_snapshot_service()
    eta = service.get_token_eta(clean_db, token_id=400, now=t_snap)
    assert eta.queue_length == 5


# =============================================================================
# TEST 5: Counter state variants
# =============================================================================
def test_05_counter_states_and_guards(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    office, services, counters = setup_standard_muni_env(clean_db, t_snap)

    # 1. Assigned + busy
    t_serv = QueueToken(
        id=501, token_number="T-SERV", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=15),
        called_at=t_snap - timedelta(minutes=5),
        serving_at=t_snap - timedelta(minutes=4),
    )
    t_wait = QueueToken(
        id=502, token_number="T-WAIT", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add_all([t_serv, t_wait])
    clean_db.commit()

    service = get_queue_snapshot_service()
    # Counter 1 busy, Counter 2 idle -> partially busy (Guard C triggers analytical fallback)
    eta_partial = service.get_token_eta(clean_db, token_id=502, now=t_snap)
    assert eta_partial.prediction_source == PredictionSource.ANALYTICAL_FALLBACK.value
    assert eta_partial.fallback_reason == FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value

    # Make Counter 2 also busy -> full capacity active and busy
    t_serv2 = QueueToken(
        id=503, token_number="T-SERV2", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=2,
        created_at=t_snap - timedelta(minutes=15),
        called_at=t_snap - timedelta(minutes=3),
        serving_at=t_snap - timedelta(minutes=2),
    )
    clean_db.add(t_serv2)
    clean_db.commit()

    # Add arrivals for in-distribution RF
    for i in range(10):
        clean_db.add(QueueToken(
            id=510 + i, token_number=f"ARR-{i}", office_id=1, service_id=1,
            status=TokenStatus.COMPLETED,
            created_at=t_snap - timedelta(minutes=5 * i + 1),
            serving_at=t_snap - timedelta(minutes=5 * i),
            completed_at=t_snap - timedelta(minutes=5 * i),
        ))
    clean_db.commit()

    eta_full = service.get_token_eta(clean_db, token_id=502, now=t_snap)
    assert eta_full.prediction_source == PredictionSource.RF.value
    assert eta_full.fallback_reason is None

    # Zero active assigned counters: mark all counters for service 1 UNAVAILABLE
    counters[0].status = CounterStatus.UNAVAILABLE
    counters[1].status = CounterStatus.UNAVAILABLE
    clean_db.commit()

    eta_zero = service.get_token_eta(clean_db, token_id=502, now=t_snap)
    assert eta_zero.prediction_source == PredictionSource.ANALYTICAL_FALLBACK.value
    assert eta_zero.fallback_reason == FallbackReason.ZERO_ACTIVE_COUNTERS.value


# =============================================================================
# TEST 6: Elapsed service time (now - service_started_at)
# =============================================================================
def test_06_elapsed_service_calculation(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Counter 1: service started 8.5 minutes ago
    clean_db.add(QueueToken(
        id=601, token_number="T1", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=20),
        called_at=t_snap - timedelta(minutes=9),
        serving_at=t_snap - timedelta(minutes=8, seconds=30),
    ))
    # Counter 2: service started 3.0 minutes ago
    clean_db.add(QueueToken(
        id=602, token_number="T2", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=2,
        created_at=t_snap - timedelta(minutes=15),
        called_at=t_snap - timedelta(minutes=4),
        serving_at=t_snap - timedelta(minutes=3),
    ))
    t_wait = QueueToken(
        id=603, token_number="T3", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=2),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, _, raw_features, _ = service.build_snapshot_and_features(clean_db, token_id=603, now=t_snap)
    # Maximum elapsed should be 8.5 minutes
    assert abs(raw_features.max_elapsed_service_minutes - 8.5) < 1e-4


# =============================================================================
# TEST 7: Arrival rates (30m / 60m calculations)
# =============================================================================
def test_07_arrival_rates(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # 4 arrivals in last 30 minutes (-25m, -20m, -15m, -5m)
    for i, m in enumerate([25, 20, 15, 5]):
        clean_db.add(QueueToken(
            id=701 + i, token_number=f"ARR-30-{i}", office_id=1, service_id=1,
            status=TokenStatus.COMPLETED,
            created_at=t_snap - timedelta(minutes=m),
        ))
    # 6 arrivals in earlier 30m of the 60m window (-55m, -50m, -45m, -40m, -35m, -32m)
    for i, m in enumerate([55, 50, 45, 40, 35, 32]):
        clean_db.add(QueueToken(
            id=710 + i, token_number=f"ARR-60-{i}", office_id=1, service_id=1,
            status=TokenStatus.COMPLETED,
            created_at=t_snap - timedelta(minutes=m),
        ))

    t_wait = QueueToken(
        id=720, token_number="T-TARGET", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=1),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, _, raw_features, _ = service.build_snapshot_and_features(clean_db, token_id=720, now=t_snap)

    # Total in 30m: 4 (arr) + 1 (target) = 5 arrivals -> rate = (5 / 30) * 60 = 10.0 arrivals/hour
    assert abs(raw_features.arrival_rate_30m - 10.0) < 1e-4
    # Total in 60m: 4 + 6 + 1 = 11 arrivals -> rate = (11 / 60) * 60 = 11.0 arrivals/hour
    assert abs(raw_features.arrival_rate_60m - 11.0) < 1e-4


# =============================================================================
# TEST 8: Point-in-time guarantee (future records excluded)
# =============================================================================
def test_08_point_in_time_guarantee(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Token at t_snap - 5m
    t_valid = QueueToken(
        id=801, token_number="T-PAST", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=5),
    )
    # Future token created 10 minutes AFTER snapshot
    t_future = QueueToken(
        id=802, token_number="T-FUTURE", office_id=1, service_id=1,
        status=TokenStatus.WAITING,
        created_at=t_snap + timedelta(minutes=10),
    )
    clean_db.add_all([t_valid, t_future])
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, snapshot, raw_features, _ = service.build_snapshot_and_features(clean_db, token_id=801, now=t_snap)

    # Future token must NOT be in waiting queue or queue length
    waiting_ids = [t.token_id for t in snapshot.queues[1].waiting_tokens]
    assert 801 in waiting_ids
    assert 802 not in waiting_ids
    assert raw_features.queue_length == 1

    # Asking ETA for future token at t_snap must fail loudly with PointInTimeViolationError
    with pytest.raises(PointInTimeViolationError):
        service.get_token_eta(clean_db, token_id=802, now=t_snap)


# =============================================================================
# TEST 9: Unsupported application identity -> HTTP 422
# =============================================================================
def test_09_unsupported_application_identity_422(clean_db, client):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)

    # Setup office RJT and service LL (known application taxonomy, but unsupported by model v2.0)
    office_rjt = Office(id=10, code="RJT", name="Rajkot Regional Office", counters=1)
    service_ll = Service(id=20, code="LL", name="Learner License", mean_minutes=12.0, std_minutes=3.0)
    clean_db.add_all([office_rjt, service_ll])
    clean_db.commit()

    counter = Counter(
        id=50, office_id=10, name="Counter 50", status=CounterStatus.ACTIVE,
        assigned_service_id=20, status_changed_at=t_snap - timedelta(hours=1),
    )
    clean_db.add(counter)
    clean_db.commit()
    clean_db.add(CounterService(counter_id=50, service_id=20))
    clean_db.commit()

    token_unsupported = QueueToken(
        id=901, token_number="RJT-LL-001", office_id=10, service_id=20,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add(token_unsupported)
    clean_db.commit()

    # Direct service call raises IdentityValidationError
    service = get_queue_snapshot_service()
    with pytest.raises(IdentityValidationError) as exc_info:
        service.get_token_eta(clean_db, token_id=901, now=t_snap)
    assert exc_info.value.code in (
        IdentityErrorCode.UNSUPPORTED_MODEL_OFFICE,
        IdentityErrorCode.UNSUPPORTED_MODEL_SERVICE,
    )

    # API call returns HTTP 422
    resp = client.get(f"/api/v1/queues/{token_unsupported.id}/eta")
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error_code"] in ["UNSUPPORTED_MODEL_OFFICE", "UNSUPPORTED_MODEL_SERVICE"]


# =============================================================================
# TEST 10: Authorization (Citizen A cannot access Citizen B's ETA)
# =============================================================================
def test_10_object_level_authorization(clean_db, client):
    app.dependency_overrides.clear()
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Users
    user_a = User(email="citizen_a@example.com", password_hash=get_password_hash("pass"), role=UserRole.citizen)
    user_b = User(email="citizen_b@example.com", password_hash=get_password_hash("pass"), role=UserRole.citizen)
    user_emp = User(email="employee@example.com", password_hash=get_password_hash("pass"), role=UserRole.employee)
    clean_db.add_all([user_a, user_b, user_emp])
    clean_db.commit()

    token_b = QueueToken(
        id=1001, token_number="MUNI-PT-101", office_id=1, service_id=1,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5),
        user_id=user_b.id,
    )
    clean_db.add(token_b)
    clean_db.commit()

    # Login helper
    def login(email):
        r = client.post("/api/v1/auth/login", json={"email": email, "password": "pass"})
        return r.json()["access_token"]

    token_auth_a = login("citizen_a@example.com")
    token_auth_b = login("citizen_b@example.com")
    token_auth_emp = login("employee@example.com")

    # 1. Citizen A requests Citizen B's token ETA -> 403 Forbidden
    resp_a = client.get(
        f"/api/v1/queues/{token_b.id}/eta",
        headers={"Authorization": f"Bearer {token_auth_a}"},
    )
    assert resp_a.status_code == 403
    assert "Forbidden" in resp_a.json()["detail"]

    # 2. Citizen B requests own token ETA -> 200 OK
    resp_b = client.get(
        f"/api/v1/queues/{token_b.id}/eta",
        headers={"Authorization": f"Bearer {token_auth_b}"},
    )
    assert resp_b.status_code == 200
    assert resp_b.json()["token_id"] == 1001

    # 3. Employee requests Citizen B's token ETA -> 200 OK (staff authorization)
    resp_emp = client.get(
        f"/api/v1/queues/{token_b.id}/eta",
        headers={"Authorization": f"Bearer {token_auth_emp}"},
    )
    assert resp_emp.status_code == 200


# =============================================================================
# TEST 11: Deterministic repeated snapshot
# =============================================================================
def test_11_deterministic_repeated_snapshot(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    t_wait = QueueToken(
        id=1101, token_number="MUNI-PT-110", office_id=1, service_id=1,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=8),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, _, feat1, _ = service.build_snapshot_and_features(clean_db, token_id=1101, now=t_snap)
    _, _, feat2, _ = service.build_snapshot_and_features(clean_db, token_id=1101, now=t_snap)

    assert feat1.to_dict() == feat2.to_dict()


# =============================================================================
# TEST 12: Exact 15-feature contract
# =============================================================================
def test_12_exact_15_feature_contract(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    t_wait = QueueToken(
        id=1201, token_number="MUNI-PT-120", office_id=1, service_id=1,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, _, raw_features, _ = service.build_snapshot_and_features(clean_db, token_id=1201, now=t_snap)

    d = raw_features.to_dict()
    assert len(d) == 15
    assert tuple(d.keys()) == RAW_FEATURE_NAMES
    for name in RAW_FEATURE_NAMES:
        assert name in d
        assert d[name] is not None


# =============================================================================
# TEST 13: HARD REQUIREMENT — Simulator / DB 15-Feature Parity Test (Section 23)
# =============================================================================
def test_13_simulator_vs_db_15_feature_parity(clean_db):
    """
    Section 23 Hard Requirement:
    At a chosen deterministic simulation moment:
    1. Obtain simulator/generator state
    2. Replay the same state into the DB using existing DB models
    3. Call the live snapshot adapter with the same now
    4. Construct the live 15-feature vector
    5. Obtain the corresponding simulator/generator feature vector
    6. Compare all 15 features (exact equality or justified float tolerance)
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)

    # 1. Simulator domain state definitions (as constructed in discrete event simulation)
    sim_services = {
        1: ServiceConfigSnapshot(
            service_id=1, service_code="PT", service_name="Property Tax",
            mean_service_minutes=10.0, std_service_minutes=4.0,
        ),
        2: ServiceConfigSnapshot(
            service_id=2, service_code="WC", service_name="Water Service",
            mean_service_minutes=15.0, std_service_minutes=6.0,
        ),
        3: ServiceConfigSnapshot(
            service_id=3, service_code="BC", service_name="Birth Certificate",
            mean_service_minutes=7.0, std_service_minutes=2.8,
        ),
    }

    # Simulation Counters:
    # Counter 1: PT, ACTIVE, busy serving Token 201 (called at -6m, serving at -5m)
    c1_serving = ServingTokenSnapshot(
        token_id=201, token_number="MUNI-PT-201", service_id=1, counter_id=1,
        status="SERVING", called_at_utc=t_snap - timedelta(minutes=6),
        serving_at_utc=t_snap - timedelta(minutes=5),
    )
    # Counter 2: PT, ACTIVE, busy serving Token 202 (called at -4m, serving at -3m)
    c2_serving = ServingTokenSnapshot(
        token_id=202, token_number="MUNI-PT-202", service_id=1, counter_id=2,
        status="SERVING", called_at_utc=t_snap - timedelta(minutes=4),
        serving_at_utc=t_snap - timedelta(minutes=3),
    )
    sim_counters = {
        1: CounterSnapshot(
            counter_id=1, office_id=1, name="Counter 1", status="ACTIVE",
            status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=c1_serving,
        ),
        2: CounterSnapshot(
            counter_id=2, office_id=1, name="Counter 2", status="ACTIVE",
            status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=c2_serving,
        ),
        3: CounterSnapshot(
            counter_id=3, office_id=1, name="Counter 3", status="ACTIVE",
            status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=2, eligible_service_ids=frozenset({2}),
            current_token=None,
        ),
    }

    # Simulation Queues:
    # Service 1 has 3 waiting tokens: Token 203 (pos 1), Token 204 (pos 2), Token 205 (pos 3)
    sim_waiting_pt = (
        QueueCitizenSnapshot(token_id=203, token_number="MUNI-PT-203", arrival_time_utc=t_snap - timedelta(minutes=15)),
        QueueCitizenSnapshot(token_id=204, token_number="MUNI-PT-204", arrival_time_utc=t_snap - timedelta(minutes=10)),
        QueueCitizenSnapshot(token_id=205, token_number="MUNI-PT-205", arrival_time_utc=t_snap - timedelta(minutes=5)),
    )
    # 8 arrival timestamps across the office/service in the 60m lookback window:
    # 3 completed tokens (at -55m, -45m, -35m) + 2 serving tokens (at -20m, -18m) + 3 waiting tokens (at -15m, -10m, -5m)
    sim_recent_pt = tuple(
        t_snap - timedelta(minutes=m) for m in [55, 45, 35, 20, 18, 15, 10, 5]
    )

    sim_queues = {
        1: ServiceQueueSnapshot(service_id=1, waiting_tokens=sim_waiting_pt, recent_arrivals_utc=sim_recent_pt),
        2: ServiceQueueSnapshot(service_id=2, waiting_tokens=(), recent_arrivals_utc=()),
        3: ServiceQueueSnapshot(service_id=3, waiting_tokens=(), recent_arrivals_utc=()),
    }

    sim_snapshot = SimulationSnapshot(
        office_id=1, office_code="MUNI", snapshot_time_utc=t_snap,
        services=sim_services, queues=sim_queues, counters=sim_counters,
    )

    # Compute simulator feature vector for Token 204 (position 2 in queue, people_ahead = 1)
    sim_features = compute_raw_features(sim_snapshot, service_id=1, position=2)

    # 2. Replay the EXACT SAME state into the database models
    office = Office(id=1, code="MUNI", name="Municipality", counters=3)
    s_pt = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=4.0)
    s_wc = Service(id=2, code="WC", name="Water Service", mean_minutes=15.0, std_minutes=6.0)
    s_bc = Service(id=3, code="BC", name="Birth Certificate", mean_minutes=7.0, std_minutes=2.8)
    clean_db.add_all([office, s_pt, s_wc, s_bc])
    clean_db.commit()

    c1 = Counter(id=1, office_id=1, name="Counter 1", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_snap - timedelta(hours=2))
    c2 = Counter(id=2, office_id=1, name="Counter 2", status=CounterStatus.ACTIVE, assigned_service_id=1, status_changed_at=t_snap - timedelta(hours=2))
    c3 = Counter(id=3, office_id=1, name="Counter 3", status=CounterStatus.ACTIVE, assigned_service_id=2, status_changed_at=t_snap - timedelta(hours=2))
    clean_db.add_all([c1, c2, c3])
    clean_db.commit()

    clean_db.add_all([
        CounterService(counter_id=1, service_id=1),
        CounterService(counter_id=2, service_id=1),
        CounterService(counter_id=3, service_id=2),
    ])
    clean_db.commit()

    # Active serving tokens in DB (created at -20m, -18m)
    clean_db.add(QueueToken(
        id=201, token_number="MUNI-PT-201", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=20),
        called_at=t_snap - timedelta(minutes=6),
        serving_at=t_snap - timedelta(minutes=5),
    ))
    clean_db.add(QueueToken(
        id=202, token_number="MUNI-PT-202", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=2,
        created_at=t_snap - timedelta(minutes=18),
        called_at=t_snap - timedelta(minutes=4),
        serving_at=t_snap - timedelta(minutes=3),
    ))

    # Waiting tokens in DB (created at -15m, -10m, -5m)
    clean_db.add(QueueToken(id=203, token_number="MUNI-PT-203", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=15)))
    clean_db.add(QueueToken(id=204, token_number="MUNI-PT-204", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=10)))
    clean_db.add(QueueToken(id=205, token_number="MUNI-PT-205", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5)))

    # Completed earlier arrivals in DB (created at -55m, -45m, -35m)
    for idx, m in enumerate([55, 45, 35]):
        clean_db.add(QueueToken(
            id=250 + idx, token_number=f"HIST-{idx}", office_id=1, service_id=1,
            status=TokenStatus.COMPLETED,
            created_at=t_snap - timedelta(minutes=m),
            completed_at=t_snap - timedelta(minutes=m - 10),
        ))
    clean_db.commit()

    # 3. Call live snapshot adapter with the same now
    service = get_queue_snapshot_service()
    _, live_snapshot, live_features, live_pos = service.build_snapshot_and_features(
        clean_db, token_id=204, now=t_snap
    )

    # 4 & 5 & 6. Verify full 15-feature parity between simulator and live DB adapter
    assert live_pos == 2

    # Feature 1: office_id
    assert live_features.office_id == sim_features.office_id == "MUNI"
    # Feature 2: service_id
    assert live_features.service_id == sim_features.service_id == "PT"
    # Feature 3: hour (in local time)
    assert live_features.hour == sim_features.hour
    # Feature 4: day_of_week
    assert live_features.day_of_week == sim_features.day_of_week
    # Feature 5: people_ahead
    assert live_features.people_ahead == sim_features.people_ahead == 1
    # Feature 6: queue_length
    assert live_features.queue_length == sim_features.queue_length == 3
    # Feature 7: assigned_active_counters
    assert live_features.assigned_active_counters == sim_features.assigned_active_counters == 2
    # Feature 8: busy_assigned_counters
    assert live_features.busy_assigned_counters == sim_features.busy_assigned_counters == 2
    # Feature 9: max_elapsed_service_minutes
    assert abs(live_features.max_elapsed_service_minutes - sim_features.max_elapsed_service_minutes) < 1e-4
    # Feature 10: service_mean_minutes
    assert abs(live_features.service_mean_minutes - sim_features.service_mean_minutes) < 1e-4
    # Feature 11: service_std_minutes
    assert abs(live_features.service_std_minutes - sim_features.service_std_minutes) < 1e-4
    # Feature 12: arrival_rate_30m
    assert abs(live_features.arrival_rate_30m - sim_features.arrival_rate_30m) < 1e-4
    # Feature 13: arrival_rate_60m
    assert abs(live_features.arrival_rate_60m - sim_features.arrival_rate_60m) < 1e-4
    # Feature 14: service_capacity_mu
    assert abs(live_features.service_capacity_mu - sim_features.service_capacity_mu) < 1e-4
    # Feature 15: traffic_intensity
    assert abs(live_features.traffic_intensity - sim_features.traffic_intensity) < 1e-4


# =============================================================================
# TEST 14: Office authority (token's stored office wins over conflicting URL)
# =============================================================================
def test_14_office_authority_mismatch_rejected(clean_db, client):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    t_wait = QueueToken(
        id=1401, token_number="MUNI-PT-140", office_id=1, service_id=1,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    # Request matching office -> 200
    resp_ok = client.get(f"/api/v1/queues/{t_wait.id}/eta", params={"office_id": 1})
    assert resp_ok.status_code == 200

    # Request with conflicting office_id=2 (token belongs to office 1) -> 400 Mismatch
    resp_bad = client.get(f"/api/v1/queues/{t_wait.id}/eta", params={"office_id": 2})
    assert resp_bad.status_code == 400
    assert "Office mismatch" in resp_bad.json()["detail"]


# =============================================================================
# TEST 15: Cold-start diagnostic
# =============================================================================
def test_15_cold_start_diagnostic(clean_db):
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Sparse DB state: exactly 1 token waiting, 0 recent arrivals in last 60m
    # arrival_rate_60m will be (1/60)*60 = 1.0 < 2.0 (below trained minimum of 2.0)
    # traffic_intensity will be 1.0 / 12.0 = 0.083 < 0.175 (below trained minimum of 0.175)
    t_wait = QueueToken(
        id=1501, token_number="MUNI-PT-150", office_id=1, service_id=1,
        status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=1),
    )
    clean_db.add(t_wait)
    clean_db.commit()

    service = get_queue_snapshot_service()
    resp = service.get_token_eta(clean_db, token_id=1501, now=t_snap)

    assert resp.is_cold_start is True
    assert len(resp.diagnostics) > 0
    diag_text = " ".join(resp.diagnostics)
    assert "ARRIVAL_RATE_60M_BELOW_TRAINED_MINIMUM" in diag_text or "TRAFFIC_INTENSITY_BELOW_TRAINED_MINIMUM" in diag_text


def _assert_15_feature_parity(live_features: RawFeatureVector, sim_features: RawFeatureVector):
    """Helper to verify exact numerical and string parity across all 15 canonical features."""
    assert live_features.office_id == sim_features.office_id
    assert live_features.service_id == sim_features.service_id
    assert live_features.hour == sim_features.hour
    assert live_features.day_of_week == sim_features.day_of_week
    assert live_features.people_ahead == sim_features.people_ahead
    assert live_features.queue_length == sim_features.queue_length
    assert live_features.assigned_active_counters == sim_features.assigned_active_counters
    assert live_features.busy_assigned_counters == sim_features.busy_assigned_counters
    assert abs(live_features.max_elapsed_service_minutes - sim_features.max_elapsed_service_minutes) < 1e-4
    assert abs(live_features.service_mean_minutes - sim_features.service_mean_minutes) < 1e-4
    assert abs(live_features.service_std_minutes - sim_features.service_std_minutes) < 1e-4
    assert abs(live_features.arrival_rate_30m - sim_features.arrival_rate_30m) < 1e-4
    assert abs(live_features.arrival_rate_60m - sim_features.arrival_rate_60m) < 1e-4
    assert abs(live_features.service_capacity_mu - sim_features.service_capacity_mu) < 1e-4
    assert abs(live_features.traffic_intensity - sim_features.traffic_intensity) < 1e-4


# =============================================================================
# TEST 16: Idle vs. Assigned Not Busy with Waiting Queue Domain Representation
# =============================================================================
def test_16_idle_vs_assigned_not_busy_with_waiting_queue_semantics(clean_db):
    """
    Demonstrates and verifies:
    1. DB Limitation:
       In the live database schema, Counter has status (ACTIVE, UNAVAILABLE), assigned_service_id,
       and status_changed_at. There is NO 'is_switching' column, NO 'SWITCHING' enum, and NO
       switching state stored in the live database schema.
       Both an idle counter and an assigned-not-busy counter are persisted as:
       `status = CounterStatus.ACTIVE` with `current_token is None`.
       The live ETA system does not claim to identify an actual switching operation.
    2. Operational / Serving distinction:
       - Idle Desk (queue_length == 0):
         An active counter with no current token when no one is waiting represents an available desk
         ready to serve an incoming citizen. Guard C does NOT fire because queue_length < 1.
       - Assigned Not Busy with Waiting Queue (queue_length >= 1):
         When citizens are waiting in queue, an active counter that is not busy (busy < assigned)
         is factually `assigned_not_busy_with_waiting_queue`.
         Guard C (PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY) safely triggers analytical fallback with
         operational capacity clamped to c = max(1, busy).
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Fetch counters from DB: Counter 1 and Counter 2 both have status ACTIVE and no tokens
    c1 = clean_db.query(Counter).filter(Counter.id == 1).first()
    c2 = clean_db.query(Counter).filter(Counter.id == 2).first()
    assert c1.status == CounterStatus.ACTIVE
    assert c2.status == CounterStatus.ACTIVE

    # Limitation check: DB records alone cannot identify an actual switching operation
    assert c1.tokens == []
    assert c2.tokens == []

    # Case A: Idle Counter with empty queue (queue_length == 0)
    # Build simulation snapshot of the live DB
    snap_idle = build_simulation_snapshot(clean_db, office_id=1, snapshot_time_utc=t_snap)
    assert snap_idle.counters[1].is_available is True
    assert snap_idle.counters[1].is_busy is False
    assert len(snap_idle.queues[1].waiting_tokens) == 0

    # Compute raw features for a hypothetical new arrival (position = 1)
    feat_idle = compute_raw_features(snap_idle, service_id=1, position=1)
    assert feat_idle.people_ahead == 0
    assert feat_idle.queue_length == 0
    assert feat_idle.assigned_active_counters == 2
    assert feat_idle.busy_assigned_counters == 0

    # Under queue_length == 0, Guard C does NOT trigger
    from app.services.prediction_service import evaluate_hybrid_serving_policy
    use_fb_idle, reason_idle = evaluate_hybrid_serving_policy(feat_idle.to_dict())
    assert use_fb_idle is False
    assert reason_idle is None

    # Case B: Assigned Not Busy with Waiting Queue (queue_length >= 1)
    # Counter 1 is actively serving a token, but Counter 2 has no token while queue has 3 waiting
    clean_db.add(QueueToken(
        id=1601, token_number="MUNI-PT-1601", office_id=1, service_id=1,
        status=TokenStatus.SERVING, counter_id=1,
        created_at=t_snap - timedelta(minutes=15),
        called_at=t_snap - timedelta(minutes=5),
        serving_at=t_snap - timedelta(minutes=4),
    ))
    clean_db.add(QueueToken(id=1602, token_number="MUNI-PT-1602", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=10)))
    clean_db.add(QueueToken(id=1603, token_number="MUNI-PT-1603", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=6)))
    clean_db.add(QueueToken(id=1604, token_number="MUNI-PT-1604", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=2)))
    clean_db.commit()

    service = get_queue_snapshot_service()
    _, snap_not_busy, feat_not_busy, _ = service.build_snapshot_and_features(clean_db, token_id=1602, now=t_snap)
    assert feat_not_busy.assigned_active_counters == 2
    assert feat_not_busy.busy_assigned_counters == 1  # 1 busy, 1 assigned_not_busy_with_waiting_queue
    assert feat_not_busy.queue_length == 3

    use_fb_sw, reason_sw = evaluate_hybrid_serving_policy(feat_not_busy.to_dict())
    assert use_fb_sw is True
    assert reason_sw == FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value


# =============================================================================
# TEST 17: Adversarial Parity Case A — Deep Queue (people_ahead >= 11)
# =============================================================================
def test_17_adversarial_parity_case_a_deep_queue(clean_db):
    """
    Verifies that for a deep queue (people_ahead >= 11):
    1. The live DB adapter and simulator construct identical 15 canonical features.
    2. The hybrid serving boundary reliably triggers Guard D (DEEP_QUEUE).
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # 2 Active serving tokens
    clean_db.add(QueueToken(
        id=1701, token_number="PT-S1", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=1, created_at=t_snap - timedelta(minutes=25), called_at=t_snap - timedelta(minutes=10), serving_at=t_snap - timedelta(minutes=8),
    ))
    clean_db.add(QueueToken(
        id=1702, token_number="PT-S2", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=2, created_at=t_snap - timedelta(minutes=22), called_at=t_snap - timedelta(minutes=8), serving_at=t_snap - timedelta(minutes=6),
    ))

    # 15 Waiting tokens: token 1713 is at position 13 (people_ahead = 12 >= 11)
    waiting_tokens_sim = []
    recent_arrivals = [t_snap - timedelta(minutes=25), t_snap - timedelta(minutes=22)]
    for i in range(1, 16):
        arr_time = t_snap - timedelta(minutes=20 - i)
        token_id = 1702 + i
        tok_num = f"PT-W{i:02d}"
        clean_db.add(QueueToken(
            id=token_id, token_number=tok_num, office_id=1, service_id=1,
            status=TokenStatus.WAITING, created_at=arr_time,
        ))
        waiting_tokens_sim.append(QueueCitizenSnapshot(token_id=token_id, token_number=tok_num, arrival_time_utc=arr_time))
        recent_arrivals.append(arr_time)

    clean_db.commit()

    # Build Simulator Snapshot
    sim_services = {
        1: ServiceConfigSnapshot(service_id=1, service_code="PT", service_name="Property Tax", mean_service_minutes=10.0, std_service_minutes=4.0),
        2: ServiceConfigSnapshot(service_id=2, service_code="WC", service_name="Water Service", mean_service_minutes=15.0, std_service_minutes=6.0),
        3: ServiceConfigSnapshot(service_id=3, service_code="BC", service_name="Birth Certificate", mean_service_minutes=7.0, std_service_minutes=2.8),
    }
    sim_counters = {
        1: CounterSnapshot(
            counter_id=1, office_id=1, name="Counter 1", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=1701, token_number="PT-S1", service_id=1, counter_id=1, status="SERVING", called_at_utc=t_snap - timedelta(minutes=10), serving_at_utc=t_snap - timedelta(minutes=8)),
        ),
        2: CounterSnapshot(
            counter_id=2, office_id=1, name="Counter 2", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=1702, token_number="PT-S2", service_id=1, counter_id=2, status="SERVING", called_at_utc=t_snap - timedelta(minutes=8), serving_at_utc=t_snap - timedelta(minutes=6)),
        ),
        3: CounterSnapshot(
            counter_id=3, office_id=1, name="Counter 3", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=2, eligible_service_ids=frozenset({2}), current_token=None,
        ),
    }
    sim_queues = {
        1: ServiceQueueSnapshot(service_id=1, waiting_tokens=tuple(waiting_tokens_sim), recent_arrivals_utc=tuple(recent_arrivals)),
        2: ServiceQueueSnapshot(service_id=2, waiting_tokens=(), recent_arrivals_utc=()),
        3: ServiceQueueSnapshot(service_id=3, waiting_tokens=(), recent_arrivals_utc=()),
    }
    sim_snapshot = SimulationSnapshot(office_id=1, office_code="MUNI", snapshot_time_utc=t_snap, services=sim_services, queues=sim_queues, counters=sim_counters)

    # Simulator features for token at position 13 (id=1715, people_ahead = 12)
    sim_features = compute_raw_features(sim_snapshot, service_id=1, position=13)

    # Live DB features for token 1715
    service = get_queue_snapshot_service()
    _, live_snap, live_features, live_pos = service.build_snapshot_and_features(clean_db, token_id=1715, now=t_snap)

    assert live_pos == 13
    assert live_features.people_ahead == 12
    assert live_features.people_ahead >= 11
    _assert_15_feature_parity(live_features, sim_features)

    # Hybrid prediction evaluates with Guard D fallback
    pred_service = get_prediction_service()
    res = pred_service.predict(live_features.to_dict())
    assert res.prediction_source == PredictionSource.ANALYTICAL_FALLBACK.value
    assert res.fallback_reason == FallbackReason.DEEP_QUEUE.value


# =============================================================================
# TEST 18: Adversarial Parity Case B — Partially Busy Counters (assigned=2, busy=1)
# =============================================================================
def test_18_adversarial_parity_case_b_partially_busy(clean_db):
    """
    Verifies that with assigned_active_counters = 2 and busy_assigned_counters = 1:
    1. The live DB adapter and simulator construct identical 15 canonical features.
    2. Guard C (PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY) fires consistently.
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Counter 1 is SERVING token 1801; Counter 2 is ACTIVE but IDLE (no token)
    clean_db.add(QueueToken(
        id=1801, token_number="PT-S1", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=1, created_at=t_snap - timedelta(minutes=15), called_at=t_snap - timedelta(minutes=6), serving_at=t_snap - timedelta(minutes=5),
    ))
    clean_db.add(QueueToken(id=1802, token_number="PT-W1", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=10)))
    clean_db.add(QueueToken(id=1803, token_number="PT-W2", office_id=1, service_id=1, status=TokenStatus.WAITING, created_at=t_snap - timedelta(minutes=5)))
    clean_db.commit()

    sim_services = {
        1: ServiceConfigSnapshot(service_id=1, service_code="PT", service_name="Property Tax", mean_service_minutes=10.0, std_service_minutes=4.0),
        2: ServiceConfigSnapshot(service_id=2, service_code="WC", service_name="Water Service", mean_service_minutes=15.0, std_service_minutes=6.0),
        3: ServiceConfigSnapshot(service_id=3, service_code="BC", service_name="Birth Certificate", mean_service_minutes=7.0, std_service_minutes=2.8),
    }
    sim_counters = {
        1: CounterSnapshot(
            counter_id=1, office_id=1, name="Counter 1", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=1801, token_number="PT-S1", service_id=1, counter_id=1, status="SERVING", called_at_utc=t_snap - timedelta(minutes=6), serving_at_utc=t_snap - timedelta(minutes=5)),
        ),
        2: CounterSnapshot(
            counter_id=2, office_id=1, name="Counter 2", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}), current_token=None,
        ),
        3: CounterSnapshot(
            counter_id=3, office_id=1, name="Counter 3", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=2, eligible_service_ids=frozenset({2}), current_token=None,
        ),
    }
    sim_waiting = (
        QueueCitizenSnapshot(token_id=1802, token_number="PT-W1", arrival_time_utc=t_snap - timedelta(minutes=10)),
        QueueCitizenSnapshot(token_id=1803, token_number="PT-W2", arrival_time_utc=t_snap - timedelta(minutes=5)),
    )
    sim_recent = (t_snap - timedelta(minutes=15), t_snap - timedelta(minutes=10), t_snap - timedelta(minutes=5))
    sim_queues = {
        1: ServiceQueueSnapshot(service_id=1, waiting_tokens=sim_waiting, recent_arrivals_utc=sim_recent),
        2: ServiceQueueSnapshot(service_id=2, waiting_tokens=(), recent_arrivals_utc=()),
        3: ServiceQueueSnapshot(service_id=3, waiting_tokens=(), recent_arrivals_utc=()),
    }
    sim_snapshot = SimulationSnapshot(office_id=1, office_code="MUNI", snapshot_time_utc=t_snap, services=sim_services, queues=sim_queues, counters=sim_counters)

    sim_features = compute_raw_features(sim_snapshot, service_id=1, position=2)

    service = get_queue_snapshot_service()
    _, live_snap, live_features, live_pos = service.build_snapshot_and_features(clean_db, token_id=1803, now=t_snap)

    assert live_pos == 2
    assert live_features.assigned_active_counters == 2
    assert live_features.busy_assigned_counters == 1
    _assert_15_feature_parity(live_features, sim_features)

    pred_service = get_prediction_service()
    res = pred_service.predict(live_features.to_dict())
    assert res.prediction_source == PredictionSource.ANALYTICAL_FALLBACK.value
    assert res.fallback_reason == FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value


# =============================================================================
# TEST 19: Adversarial Parity Case C — Boundary (people_ahead = 10 vs 11)
# =============================================================================
def test_19_adversarial_parity_case_c_boundary_10_vs_11(clean_db):
    """
    Verifies feature parity and boundary behavior at:
    - people_ahead = 10 (Position 11 in waiting queue) -> Evaluates via RF model
    - people_ahead = 11 (Position 12 in waiting queue) -> Evaluates via Guard D fallback
    Proves feature vectors match identically between simulation and live DB across the threshold.
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # Both assigned counters are busy
    clean_db.add(QueueToken(
        id=1901, token_number="PT-S1", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=1, created_at=t_snap - timedelta(minutes=25), called_at=t_snap - timedelta(minutes=10), serving_at=t_snap - timedelta(minutes=8),
    ))
    clean_db.add(QueueToken(
        id=1902, token_number="PT-S2", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=2, created_at=t_snap - timedelta(minutes=22), called_at=t_snap - timedelta(minutes=8), serving_at=t_snap - timedelta(minutes=6),
    ))

    waiting_sim = []
    recent_arrivals = [t_snap - timedelta(minutes=25), t_snap - timedelta(minutes=22)]
    # Add 14 waiting tokens
    for i in range(1, 15):
        arr_time = t_snap - timedelta(minutes=20 - i)
        tok_id = 1902 + i
        tok_num = f"PT-W{i:02d}"
        clean_db.add(QueueToken(
            id=tok_id, token_number=tok_num, office_id=1, service_id=1,
            status=TokenStatus.WAITING, created_at=arr_time,
        ))
        waiting_sim.append(QueueCitizenSnapshot(token_id=tok_id, token_number=tok_num, arrival_time_utc=arr_time))
        recent_arrivals.append(arr_time)

    clean_db.commit()

    sim_services = {
        1: ServiceConfigSnapshot(service_id=1, service_code="PT", service_name="Property Tax", mean_service_minutes=10.0, std_service_minutes=4.0),
        2: ServiceConfigSnapshot(service_id=2, service_code="WC", service_name="Water Service", mean_service_minutes=15.0, std_service_minutes=6.0),
        3: ServiceConfigSnapshot(service_id=3, service_code="BC", service_name="Birth Certificate", mean_service_minutes=7.0, std_service_minutes=2.8),
    }
    sim_counters = {
        1: CounterSnapshot(
            counter_id=1, office_id=1, name="Counter 1", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=1901, token_number="PT-S1", service_id=1, counter_id=1, status="SERVING", called_at_utc=t_snap - timedelta(minutes=10), serving_at_utc=t_snap - timedelta(minutes=8)),
        ),
        2: CounterSnapshot(
            counter_id=2, office_id=1, name="Counter 2", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=1902, token_number="PT-S2", service_id=1, counter_id=2, status="SERVING", called_at_utc=t_snap - timedelta(minutes=8), serving_at_utc=t_snap - timedelta(minutes=6)),
        ),
        3: CounterSnapshot(
            counter_id=3, office_id=1, name="Counter 3", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=2, eligible_service_ids=frozenset({2}), current_token=None,
        ),
    }
    sim_queues = {
        1: ServiceQueueSnapshot(service_id=1, waiting_tokens=tuple(waiting_sim), recent_arrivals_utc=tuple(recent_arrivals)),
        2: ServiceQueueSnapshot(service_id=2, waiting_tokens=(), recent_arrivals_utc=()),
        3: ServiceQueueSnapshot(service_id=3, waiting_tokens=(), recent_arrivals_utc=()),
    }
    sim_snapshot = SimulationSnapshot(office_id=1, office_code="MUNI", snapshot_time_utc=t_snap, services=sim_services, queues=sim_queues, counters=sim_counters)

    service = get_queue_snapshot_service()
    pred_service = get_prediction_service()

    # 1. Test position 11 (people_ahead = 10): id=1913
    sim_feat_10 = compute_raw_features(sim_snapshot, service_id=1, position=11)
    _, _, live_feat_10, live_pos_10 = service.build_snapshot_and_features(clean_db, token_id=1913, now=t_snap)
    assert live_pos_10 == 11
    assert live_feat_10.people_ahead == 10
    _assert_15_feature_parity(live_feat_10, sim_feat_10)
    res_10 = pred_service.predict(live_feat_10.to_dict())
    assert res_10.prediction_source == PredictionSource.RF.value

    # 2. Test position 12 (people_ahead = 11): id=1914
    sim_feat_11 = compute_raw_features(sim_snapshot, service_id=1, position=12)
    _, _, live_feat_11, live_pos_11 = service.build_snapshot_and_features(clean_db, token_id=1914, now=t_snap)
    assert live_pos_11 == 12
    assert live_feat_11.people_ahead == 11
    _assert_15_feature_parity(live_feat_11, sim_feat_11)
    res_11 = pred_service.predict(live_feat_11.to_dict())
    assert res_11.prediction_source == PredictionSource.ANALYTICAL_FALLBACK.value
    assert res_11.fallback_reason == FallbackReason.DEEP_QUEUE.value


# =============================================================================
# TEST 20: Adversarial Parity Case D — Cold Start
# =============================================================================
def test_20_adversarial_parity_case_d_cold_start(clean_db):
    """
    Verifies that with zero recent arrival history in trailing 30m/60m lookback windows:
    1. The live DB adapter and simulator construct identical 15 canonical features
       (arrival_rate_30m=0.0, arrival_rate_60m=0.0, traffic_intensity=0.0).
    2. The live ETA response reports is_cold_start=True with diagnostic tags.
    """
    t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=timezone.utc)
    setup_standard_muni_env(clean_db, t_snap)

    # 2 Active serving tokens created 10 seconds ago (outside 30m lookback window? No, inside 30m lookback window!)
    # To test zero recent arrival history, create NO tokens in the lookback window except the target token created at t_snap
    # 2 Active serving tokens created 90 minutes ago (outside the 60m window: > 60 min ago)
    clean_db.add(QueueToken(
        id=2001, token_number="PT-S1", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=1, created_at=t_snap - timedelta(minutes=90), called_at=t_snap - timedelta(minutes=10), serving_at=t_snap - timedelta(minutes=8),
    ))
    clean_db.add(QueueToken(
        id=2002, token_number="PT-S2", office_id=1, service_id=1, status=TokenStatus.SERVING,
        counter_id=2, created_at=t_snap - timedelta(minutes=85), called_at=t_snap - timedelta(minutes=8), serving_at=t_snap - timedelta(minutes=6),
    ))

    # Single waiting token created exactly at t_snap (so in (t_snap - 60m, t_snap], count is 0 if exclusive, or 1)
    # If waiting token was created at t_snap - 65m: then 0 arrivals in (t_snap - 60m, t_snap]
    clean_db.add(QueueToken(
        id=2003, token_number="PT-W1", office_id=1, service_id=1, status=TokenStatus.WAITING,
        created_at=t_snap - timedelta(minutes=65),
    ))
    clean_db.commit()

    sim_services = {
        1: ServiceConfigSnapshot(service_id=1, service_code="PT", service_name="Property Tax", mean_service_minutes=10.0, std_service_minutes=4.0),
        2: ServiceConfigSnapshot(service_id=2, service_code="WC", service_name="Water Service", mean_service_minutes=15.0, std_service_minutes=6.0),
        3: ServiceConfigSnapshot(service_id=3, service_code="BC", service_name="Birth Certificate", mean_service_minutes=7.0, std_service_minutes=2.8),
    }
    sim_counters = {
        1: CounterSnapshot(
            counter_id=1, office_id=1, name="Counter 1", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=2001, token_number="PT-S1", service_id=1, counter_id=1, status="SERVING", called_at_utc=t_snap - timedelta(minutes=10), serving_at_utc=t_snap - timedelta(minutes=8)),
        ),
        2: CounterSnapshot(
            counter_id=2, office_id=1, name="Counter 2", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=1, eligible_service_ids=frozenset({1}),
            current_token=ServingTokenSnapshot(token_id=2002, token_number="PT-S2", service_id=1, counter_id=2, status="SERVING", called_at_utc=t_snap - timedelta(minutes=8), serving_at_utc=t_snap - timedelta(minutes=6)),
        ),
        3: CounterSnapshot(
            counter_id=3, office_id=1, name="Counter 3", status="ACTIVE", status_changed_at_utc=t_snap - timedelta(hours=2),
            assigned_service_id=2, eligible_service_ids=frozenset({2}), current_token=None,
        ),
    }
    # No arrivals in last 60 minutes
    sim_queues = {
        1: ServiceQueueSnapshot(
            service_id=1,
            waiting_tokens=(QueueCitizenSnapshot(token_id=2003, token_number="PT-W1", arrival_time_utc=t_snap - timedelta(minutes=65)),),
            recent_arrivals_utc=(),
        ),
        2: ServiceQueueSnapshot(service_id=2, waiting_tokens=(), recent_arrivals_utc=()),
        3: ServiceQueueSnapshot(service_id=3, waiting_tokens=(), recent_arrivals_utc=()),
    }
    sim_snapshot = SimulationSnapshot(office_id=1, office_code="MUNI", snapshot_time_utc=t_snap, services=sim_services, queues=sim_queues, counters=sim_counters)

    sim_features = compute_raw_features(sim_snapshot, service_id=1, position=1)

    service = get_queue_snapshot_service()
    _, live_snap, live_features, live_pos = service.build_snapshot_and_features(clean_db, token_id=2003, now=t_snap)

    assert live_pos == 1
    assert live_features.arrival_rate_30m == 0.0
    assert live_features.arrival_rate_60m == 0.0
    assert live_features.traffic_intensity == 0.0
    _assert_15_feature_parity(live_features, sim_features)

    # Verify diagnostic flags
    eta_resp = service.get_token_eta(clean_db, token_id=2003, now=t_snap)
    assert eta_resp.is_cold_start is True
    assert "ARRIVAL_RATE_60M_BELOW_TRAINED_MINIMUM" in " ".join(eta_resp.diagnostics)

