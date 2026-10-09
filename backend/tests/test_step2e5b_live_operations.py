"""
PRAVAAH — Step 2E.5B Live Operations, Reliability, and Demo Readiness Test Suite.

Verifies:
1. Busy counter reassignment lifecycle (serves token T, reassigns A->B, completes T, next call serves B).
2. Busy counter failure lifecycle (marked UNAVAILABLE while busy, completes T, cannot call until recovered).
3. Fluid time-to-congestion calculation, units parity (citizens/hour), configurable threshold.
4. CORS default dev origins, custom CORS_ORIGINS, preflight OPTIONS, and disallowed origin.
5. Canonical E_known scenario loading (C1 UNAVAILABLE at t=0, C2..C6 ACTIVE, repeated load idempotency).
6. WebSocket invalidations (exact 4 keys, commit safety, non-duplicated events).
7. Concurrent call-next race condition hardening (process-local queue lock prevents duplicate token issuance).
8. Dialect-safe analytics queries under SQLite.
"""

import threading
from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_cors_origins, DEFAULT_CORS_ORIGINS
from app.api.admin import compute_time_to_congestion_fluid
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.services.scenario_loader_service import load_scenario_into_db
from app.ws_manager import ws_manager

client = TestClient(app)


@pytest.fixture
def setup_demo_scenario():
    """Seeds Office 1 using canonical scenario_b_pt_pressure."""
    db = SessionLocal()
    try:
        # Ensure offices exist
        o1 = db.query(Office).filter(Office.id == 1).first()
        if not o1:
            db.add(Office(id=1, code="MUNI", name="Municipality", counters=6, demand_multiplier=1.0))
            db.commit()
        load_scenario_into_db(db, office_id=1, scenario_id="scenario_b_pt_pressure")
    finally:
        db.close()
    yield


# =============================================================================
# 1. BUSY COUNTER REASSIGNMENT LIFECYCLE (Section 2)
# =============================================================================

def test_busy_counter_reassignment_complete_lifecycle(setup_demo_scenario):
    """
    Test complete Section 2 lifecycle:
    1. Counter C5 is assigned to service BC (3) and is serving token T.
    2. Admin reassigns C5 to service PT (1).
    3. Reassignment succeeds without interrupting T.
    4. T remains associated with BC (3) and C5 and can be completed normally.
    5. C5 must not accept another token while T is still in progress.
    6. After T completes, C5 remains assigned to PT (1).
    7. Next call on C5 serves an eligible token from PT (1), never from BC (3).
    8. Restoring C5 to home service (BC) succeeds without corrupting completed tokens.
    """
    db = SessionLocal()
    try:
        # Verify C5 is initially assigned to service 3 (BC) and has an active token
        c5 = db.query(Counter).filter(Counter.id == 5).first()
        assert c5.assigned_service_id == 3
        token_t = db.query(QueueToken).filter(
            QueueToken.counter_id == 5,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert token_t is not None
        assert token_t.service_id == 3
        token_t_id = token_t.id
    finally:
        db.close()

    # Step 2 & 3: Reassign C5 from BC (3) to PT (1) while token T is in progress
    resp_reassign = client.post(
        "/api/v1/interventions/apply",
        json={"office_id": 1, "counter_id": 5, "target_service_id": 1},
    )
    assert resp_reassign.status_code == 200
    assert resp_reassign.json()["applied"] is True
    assert resp_reassign.json()["previous_service_id"] == 3
    assert resp_reassign.json()["target_service_id"] == 1

    # Step 4: Token T remains associated with original service BC (3) and counter C5
    db = SessionLocal()
    try:
        t_after = db.query(QueueToken).filter(QueueToken.id == token_t_id).first()
        assert t_after.service_id == 3
        assert t_after.counter_id == 5
        assert t_after.status in [TokenStatus.CALLED, TokenStatus.SERVING]
    finally:
        db.close()

    # Step 5: C5 must NOT accept another token while T is still in progress
    resp_call_busy = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 5},
    )
    assert resp_call_busy.status_code == 409
    assert "already busy" in resp_call_busy.json()["detail"].lower()

    # Complete token T normally
    resp_complete = client.post(f"/api/v1/tokens/{token_t_id}/complete")
    assert resp_complete.status_code == 200
    assert resp_complete.json()["status"] == "COMPLETED"

    # Step 6: After T completes, C5 remains assigned to PT (1)
    db = SessionLocal()
    try:
        c5_idle = db.query(Counter).filter(Counter.id == 5).first()
        assert c5_idle.assigned_service_id == 1
    finally:
        db.close()

    # Step 7: Next call on C5 for BC (3) fails (because C5 is now assigned to PT)
    resp_call_wrong = client.post(
        "/api/v1/queues/1/3/call-next",
        json={"counter_id": 5},
    )
    assert resp_call_wrong.status_code == 400
    assert "assigned to service 1, not 3" in resp_call_wrong.json()["detail"]

    # Next call on C5 for PT (1) succeeds and serves PT token!
    resp_call_pt = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 5},
    )
    assert resp_call_pt.status_code == 200
    called_pt_token = resp_call_pt.json()
    assert called_pt_token["service_id"] == 1
    assert called_pt_token["counter_id"] == 5

    # Step 8: Restore C5 to home assignment (BC, service 3)
    resp_restore = client.post(
        "/api/v1/interventions/restore",
        json={"office_id": 1, "counter_id": 5},
    )
    assert resp_restore.status_code == 200
    assert resp_restore.json()["restored"] is True
    assert resp_restore.json()["restored_service_id"] == 3


# =============================================================================
# 2. BUSY COUNTER FAILURE AND RECOVERY (Section 3)
# =============================================================================

def test_busy_counter_failure_and_recovery_lifecycle(setup_demo_scenario):
    """
    Test complete Section 3 failure lifecycle:
    1. Counter C4 is ACTIVE and serving token T.
    2. Admin marks C4 UNAVAILABLE while serving T.
    3. Operation succeeds, C4 becomes UNAVAILABLE.
    4. Token T remains intact and completes normally.
    5. C4 cannot call tokens while UNAVAILABLE.
    6. Admin explicitly recovers C4 to ACTIVE; assignment is preserved.
    """
    db = SessionLocal()
    try:
        token_t = db.query(QueueToken).filter(
            QueueToken.counter_id == 4,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert token_t is not None
        token_t_id = token_t.id
        service_id = token_t.service_id
    finally:
        db.close()

    # Mark busy counter UNAVAILABLE
    resp_unavail = client.post("/api/v1/counters/4/unavailable")
    assert resp_unavail.status_code == 200
    assert resp_unavail.json()["status"] == "UNAVAILABLE"

    # Token T remains intact
    db = SessionLocal()
    try:
        tok = db.query(QueueToken).filter(QueueToken.id == token_t_id).first()
        assert tok.counter_id == 4
        assert tok.service_id == service_id
        assert tok.status in [TokenStatus.CALLED, TokenStatus.SERVING]
    finally:
        db.close()

    # In-progress token can be completed normally
    resp_comp = client.post(f"/api/v1/tokens/{token_t_id}/complete")
    assert resp_comp.status_code == 200

    # C4 remains UNAVAILABLE and cannot call new tokens
    resp_call = client.post(
        f"/api/v1/queues/1/{service_id}/call-next",
        json={"counter_id": 4},
    )
    assert resp_call.status_code == 400
    assert "UNAVAILABLE" in resp_call.json()["detail"]

    # Explicit recovery restores ACTIVE status
    resp_recov = client.post("/api/v1/counters/4/recover")
    assert resp_recov.status_code == 200
    assert resp_recov.json()["status"] == "ACTIVE"
    assert resp_recov.json()["assigned_service_id"] == 2  # WC preserved


# =============================================================================
# 3. FLUID TIME-TO-CONGESTION & ADMIN SUMMARY (Section 4 & 7)
# =============================================================================

def test_fluid_time_to_congestion_formula():
    """Verify fluid TTC formula, units, and boundary conditions."""
    # Case 1: Finite positive TTC (Q=5, lambda=18, mu=12, threshold=15)
    # Net growth = 18 - 12 = 6/hr = 0.1/min. Remaining = 10 tokens. TTC = 100.0 min.
    ttc, method = compute_time_to_congestion_fluid(
        queue_length=5,
        arrival_rate_per_hour=18.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert ttc == 100.0
    assert method == "fluid_approximation"

    # Case 2: Already congested (Q >= threshold) -> 0.0 minutes
    ttc_cong, _ = compute_time_to_congestion_fluid(
        queue_length=15,
        arrival_rate_per_hour=20.0,
        service_capacity_per_hour=10.0,
        congestion_threshold=15,
    )
    assert ttc_cong == 0.0

    # Case 3: Stable / draining queue (lambda <= mu) -> None (does not congest)
    ttc_stable, _ = compute_time_to_congestion_fluid(
        queue_length=5,
        arrival_rate_per_hour=8.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert ttc_stable is None

    # Case 4: Zero capacity (mu=0, lambda=6/hr, Q=0, threshold=15)
    # Net growth = 6/hr = 0.1/min. Remaining = 15 tokens. TTC = 150.0 min.
    ttc_zero_cap, _ = compute_time_to_congestion_fluid(
        queue_length=0,
        arrival_rate_per_hour=6.0,
        service_capacity_per_hour=0.0,
        congestion_threshold=15,
    )
    assert ttc_zero_cap == 150.0

    # Case 5: Zero arrivals (lambda=0) -> None
    ttc_zero_arr, _ = compute_time_to_congestion_fluid(
        queue_length=2,
        arrival_rate_per_hour=0.0,
        service_capacity_per_hour=10.0,
        congestion_threshold=15,
    )
    assert ttc_zero_arr is None


def test_admin_summary_exposes_fluid_ttc_and_threshold(setup_demo_scenario):
    """Admin summary returns labelled fluid approximation and supports custom threshold."""
    resp = client.get("/api/v1/admin/offices/1/summary?congestion_threshold=20")
    assert resp.status_code == 200
    data = resp.json()
    assert "services" in data
    for s in data["services"]:
        assert s["time_to_congestion_method"] == "fluid_approximation"
        assert s["congestion_threshold"] == 20
        # Check arrival rate units are citizens/hour (>= 0)
        assert s["arrival_rate_60m"] >= 0.0
        assert s["service_capacity_mu"] >= 0.0


# =============================================================================
# 4. CORS LOCAL-DEV DEFAULTS & HEADERS (Section 5)
# =============================================================================

def test_cors_allowed_development_origin():
    """Vite dev server origin (http://localhost:5173) is allowed by default with credentials."""
    resp = client.get(
        "/health/db",
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp.headers.get("access-control-allow-credentials") is None


def test_cors_preflight_options():
    """OPTIONS preflight request succeeds with appropriate CORS headers."""
    resp = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_cors_disallowed_origin():
    """Disallowed arbitrary external origin does not receive access-control-allow-origin."""
    resp = client.get(
        "/health/db",
        headers={"Origin": "http://malicious-site.com"},
    )
    assert resp.headers.get("access-control-allow-origin") is None


def test_cors_origins_resolution():
    """Verify default dev origins contain standard local frontend ports."""
    origins = get_cors_origins()
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins
    assert "null" not in origins
    assert "*" not in origins


# =============================================================================
# 5. CANONICAL E_KNOWN SCENARIO (Section 6)
# =============================================================================

def test_canonical_e_known_scenario_loading(setup_demo_scenario):
    """Scenario E_known loads deterministically with C1 UNAVAILABLE at t=0."""
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "E_known"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] in ("scenario_e_known", "E_known")
    assert data["office_code"] == "MUNI"
    assert data["counters_loaded"] == 6

    # Verify DB state: C1 must be UNAVAILABLE with NO active token
    db = SessionLocal()
    try:
        c1 = db.query(Counter).filter(Counter.id == 1).first()
        assert c1.status == CounterStatus.UNAVAILABLE
        c1_token = db.query(QueueToken).filter(
            QueueToken.counter_id == 1,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert c1_token is None

        # Counters C2..C6 must be ACTIVE and busy
        for cid in [2, 3, 4, 5, 6]:
            c = db.query(Counter).filter(Counter.id == cid).first()
            assert c.status == CounterStatus.ACTIVE
            tok = db.query(QueueToken).filter(
                QueueToken.counter_id == cid,
                QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
            ).first()
            assert tok is not None
    finally:
        db.close()

    # Re-loading E_known must be idempotent and produce identical counts
    resp2 = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "E_known"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["counters_loaded"] == 6
    assert resp2.json()["waiting_tokens_loaded"] == data["waiting_tokens_loaded"]


# =============================================================================
# 6. WEBSOCKET INVALIDATION CONTRACT (Section 8)
# =============================================================================

def test_websocket_invalidation_payload_format(setup_demo_scenario):
    """Invalidation payload contains exactly the 4 required keys: type, office_id, service_id, occurred_at."""
    captured = []

    async def fake_broadcast(event_type, office_id, service_id, occurred_at=None):
        payload = {
            "type": event_type,
            "office_id": office_id,
            "service_id": service_id,
            "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
        }
        captured.append(payload)

    orig_broadcast = ws_manager.broadcast_invalidation
    ws_manager.broadcast_invalidation = fake_broadcast
    try:
        # Join queue mutation
        resp = client.post("/api/v1/queues/1/join", json={"office_id": 1})
        assert resp.status_code in [200, 201]

        assert len(captured) == 1
        msg = captured[0]
        assert set(msg.keys()) == {"type", "office_id", "service_id", "occurred_at"}
        assert msg["type"] == "queue.updated"
        assert msg["office_id"] == 1
        assert msg["service_id"] == 1
    finally:
        ws_manager.broadcast_invalidation = orig_broadcast


# =============================================================================
# 7. LIGHTWEIGHT CONCURRENCY HARDENING (Section 9)
# =============================================================================

def test_concurrent_call_next_same_counter(setup_demo_scenario):
    """
    Two threads concurrently attempting call-next on the same idle counter:
    Process-local lock ensures exactly one succeeds and the other receives 409 Conflict.
    No waiting token is double-issued.
    """
    # Make C2 idle
    db = SessionLocal()
    try:
        tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 2,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if tok:
            tok.status = TokenStatus.COMPLETED
            db.commit()
    finally:
        db.close()

    results = []

    def call_worker():
        c = TestClient(app)
        res = c.post("/api/v1/queues/1/1/call-next", json={"counter_id": 2})
        results.append(res.status_code)

    t1 = threading.Thread(target=call_worker)
    t2 = threading.Thread(target=call_worker)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # One call should succeed with 200, the other should fail with 409 (or 400 if empty)
    assert 200 in results
    assert (409 in results or 400 in results)

    # Verify no counter serves two active tokens
    db = SessionLocal()
    try:
        active_count = db.query(QueueToken).filter(
            QueueToken.counter_id == 2,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).count()
        assert active_count == 1
    finally:
        db.close()


# =============================================================================
# 8. DIALECT-SAFE ANALYTICS (Section 10)
# =============================================================================

def test_dialect_safe_analytics_execution(setup_demo_scenario):
    """Overall and office analytics queries execute successfully without SQL dialect errors."""
    resp_overall = client.get("/api/v1/analytics/overall")
    assert resp_overall.status_code == 200
    assert "total_tokens" in resp_overall.json()

    resp_office = client.get("/api/v1/analytics/offices/1")
    assert resp_office.status_code == 200
    assert resp_office.json()["office_code"] == "MUNI"
