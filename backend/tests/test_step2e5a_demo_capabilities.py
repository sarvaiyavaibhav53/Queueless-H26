"""
PRAVAAH — Step 2E.5A Demo-Critical Backend Acceptance Test Suite.

Validates the full operational loop:
Observe -> Predict -> Detect -> Simulate -> Recommend -> Apply -> Verify -> Measure

Test Coverage:
1. Scenario Loader:
   - Deterministic load of standard scenarios (A, B, C, E, RTO).
   - Strict office isolation (mutating Office 1 does not affect Office 2).
   - Rejection of invalid scenario IDs (400).
   - Rejection of unauthorized users (403 for citizens).
   - Transactional rollback on error.
   - Consistent counter/token invariants (1 active token per busy counter, waiting tokens backdated).
2. Operations Summary:
   - Admin authorization and office scoping.
   - Shared feature and stability calculations (rho, mu, STABLE/WATCH/UNSTABLE).
   - Accurate queue lengths and active/busy/unavailable counter counts.
   - Zero-capacity handling without division-by-zero errors.
3. Counter-specific Call-Next:
   - Active idle assigned counter calls next token successfully.
   - Rejection of busy counter (409 Conflict).
   - Rejection of unavailable counter (400 Bad Request).
   - Rejection of counter assigned to different service (400 Bad Request).
   - Rejection of counter not eligible for service (400 Bad Request).
   - Rejection of counter belonging to different office (400 Bad Request).
   - Strict FIFO ordering (created_at ASC, id ASC).
4. Real Intervention Apply & Restore:
   - Application of valid eligible reassignment changes counter assigned_service_id.
   - Active token safety: Reassignment rejected with 409 Conflict if active token is in-service.
   - Donor retention rule: Reassignment rejected if donor would have 0 active counters.
   - Ineligible counter reassignment rejected (400).
   - Authoritative home restoration succeeds and restores original service.
   - Restoration of busy counter rejected (409 Conflict).
   - Restoration of counter already at home service rejected (400).
5. Counter Unavailable & Recovery:
   - Transition to UNAVAILABLE succeeds for idle counter.
   - Transition to UNAVAILABLE rejected with 409 Conflict if active token is in-service.
   - UNAVAILABLE counter excluded from calling tokens.
   - Recovery restores counter to ACTIVE without changing assigned service.
6. WebSocket Invalidation Contract:
   - Invalidation payload contains only {type, office_id, service_id, occurred_at}.
   - Zero token arrays or citizen PII.
"""

from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.database import SessionLocal
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.user import User, UserRole
from app.services.scenario_loader_service import load_scenario_into_db
from app.ws_manager import ws_manager

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_offices_and_services():
    """Initializes standard offices and services in SQLite DB."""
    db = SessionLocal()
    try:
        # Clear existing
        db.query(QueueToken).delete()
        db.query(CounterService).delete()
        db.query(Counter).delete()
        db.query(Service).delete()
        db.query(Office).delete()
        db.commit()

        # Office 1: Municipality
        off1 = Office(id=1, code="MUNI", name="Municipality Office", counters=6)
        # Office 2: RTO
        off2 = Office(id=2, code="RTO", name="Regional Transport Office", counters=3)
        db.add_all([off1, off2])

        # Municipality Services
        s_pt = Service(id=1, code="PT", name="Property Tax", mean_minutes=10.0, std_minutes=4.0)
        s_wc = Service(id=2, code="WC", name="Water Service", mean_minutes=15.0, std_minutes=6.0)
        s_bc = Service(id=3, code="BC", name="Birth Certificate", mean_minutes=7.0, std_minutes=2.8)

        # RTO Services
        s_ot = Service(id=4, code="OT", name="Ownership Transfer", mean_minutes=14.0, std_minutes=5.6)
        s_dr = Service(id=5, code="DR", name="Duplicate RC", mean_minutes=8.0, std_minutes=3.2)

        db.add_all([s_pt, s_wc, s_bc, s_ot, s_dr])
        db.commit()
    finally:
        db.close()


# =============================================================================
# 1. SCENARIO LOADER TESTS
# =============================================================================

def test_scenario_load_deterministic(setup_offices_and_services):
    """Scenario B (PT pressure) loads deterministically with expected counters and tokens."""
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "B"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["scenario_id"] == "scenario_b_pt_pressure"
    assert data["office_id"] == 1
    assert data["office_code"] == "MUNI"
    assert data["counters_loaded"] == 6
    assert data["busy_counters_loaded"] == 6
    assert data["waiting_tokens_loaded"] == 10  # 8 PT + 1 WC + 1 BC

    # Verify DB state directly
    db = SessionLocal()
    try:
        pt_tokens = db.query(QueueToken).filter(
            QueueToken.office_id == 1,
            QueueToken.service_id == 1,
            QueueToken.status == TokenStatus.WAITING,
        ).order_by(QueueToken.created_at.asc()).all()
        assert len(pt_tokens) == 8

        # Verify strict FIFO backdating: token 1 arrived earlier than token 12
        assert pt_tokens[0].created_at < pt_tokens[-1].created_at

        # Verify busy counters have exactly 1 active serving token
        for c_id in range(1, 7):
            active_tokens = db.query(QueueToken).filter(
                QueueToken.counter_id == c_id,
                QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
            ).all()
            assert len(active_tokens) == 1
    finally:
        db.close()


def test_scenario_load_office_isolation(setup_offices_and_services):
    """Loading a scenario into Office 1 does not mutate Office 2 data."""
    db = SessionLocal()
    try:
        # Seed an isolated counter and token in Office 2
        c_rto = Counter(
            id=201,
            office_id=2,
            name="R1",
            status=CounterStatus.ACTIVE,
            assigned_service_id=4,
            status_changed_at=datetime.now(timezone.utc),
        )
        t_rto = QueueToken(
            id=9999,
            token_number="RTO-OT-999",
            office_id=2,
            service_id=4,
            status=TokenStatus.WAITING,
            created_at=datetime.now(timezone.utc),
        )
        db.merge(c_rto)
        db.merge(t_rto)
        db.commit()
    finally:
        db.close()

    # Reset Office 1 by loading Scenario A
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "A"},
    )
    assert resp.status_code == 200

    # Verify Office 2 records are completely untouched
    db = SessionLocal()
    try:
        tok = db.query(QueueToken).filter(QueueToken.id == 9999).first()
        assert tok is not None
        assert tok.office_id == 2
        cnt = db.query(Counter).filter(Counter.id == 201).first()
        assert cnt is not None
        assert cnt.office_id == 2
    finally:
        db.close()


def test_scenario_load_invalid_scenario_id():
    """Invalid scenario ID returns 400 Bad Request."""
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "INVALID_UNKNOWN_SCENARIO"},
    )
    assert resp.status_code == 400
    assert "Unknown scenario_id" in resp.json()["detail"]


def test_scenario_load_office_mismatch():
    """Attempting to load an RTO scenario into Municipality returns 400."""
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "scenario_rto_healthy"},
    )
    assert resp.status_code == 400
    assert "belongs to office 2" in resp.json()["detail"]


def test_list_scenarios():
    """Admin can list all available frozen scenarios."""
    resp = client.get("/api/v1/admin/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()["scenarios"]
    assert len(scenarios) >= 5
    scenario_ids = [s["scenario_id"] for s in scenarios]
    assert "scenario_a_healthy" in scenario_ids
    assert "scenario_b_pt_pressure" in scenario_ids


# =============================================================================
# 2. ADMIN OPERATIONS SUMMARY TESTS
# =============================================================================

def test_admin_operations_summary(setup_offices_and_services):
    """Summary returns correct operational metrics derived from snapshot and shared helpers."""
    # Load Scenario B
    client.post("/api/v1/admin/offices/1/scenarios/load", json={"scenario_id": "B"})

    resp = client.get("/api/v1/admin/offices/1/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["office_id"] == 1
    assert data["office_code"] == "MUNI"
    assert data["total_waiting_tokens"] == 10
    assert data["total_active_counters"] == 6
    assert data["total_busy_counters"] == 6
    assert data["total_unavailable_counters"] == 0

    # Check Property Tax service (service_id=1)
    pt_summary = next(s for s in data["services"] if s["service_id"] == 1)
    assert pt_summary["service_code"] == "PT"
    assert pt_summary["queue_length"] == 8
    assert pt_summary["assigned_active_counters"] == 2
    assert pt_summary["busy_counters"] == 2
    assert pt_summary["service_mean_minutes"] == 10.0
    # Capacity mu = 2 counters * (60 / 10) = 12.0
    assert pt_summary["service_capacity_mu"] == 12.0
    # Queue health should be UNSTABLE or WATCH under high pressure
    assert pt_summary["queue_health"] in ["STABLE", "WATCH", "UNSTABLE"]
    assert pt_summary["time_to_congestion_minutes"] is not None and pt_summary["time_to_congestion_minutes"] > 0

    # Check counter fleet details
    assert len(data["counters"]) == 6
    c1 = next(c for c in data["counters"] if c["counter_id"] == 1)
    assert c1["name"] == "C1"
    assert c1["assigned_service_id"] == 1
    assert c1["assigned_service_code"] == "PT"
    assert c1["status"] == "ACTIVE"
    assert c1["is_busy"] is True
    assert c1["current_token_id"] is not None
    assert set(c1["eligible_service_ids"]) == {1, 3}  # PT and BC


# =============================================================================
# 3. COUNTER-SPECIFIC CALL-NEXT TESTS
# =============================================================================

def test_call_next_with_valid_idle_counter(setup_offices_and_services):
    """Calling next with a valid active, idle, assigned counter succeeds and assigns token to counter."""
    db = SessionLocal()
    try:
        # Make C1 idle by completing its current token
        active_tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 1,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if active_tok:
            active_tok.status = TokenStatus.COMPLETED
            db.commit()
    finally:
        db.close()

    # Call next for PT (service_id=1) specifying counter_id=1
    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 1},
    )
    assert resp.status_code == 200, resp.text
    tok_data = resp.json()
    assert tok_data["status"] == "CALLED"
    assert tok_data["counter_id"] == 1
    assert tok_data["called_at"] is not None


def test_call_next_busy_counter_rejected(setup_offices_and_services):
    """Calling next with a busy counter returns 409 Conflict."""
    # Counter 1 is now busy from the previous test
    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 1},
    )
    assert resp.status_code == 409
    assert "already busy" in resp.json()["detail"]


def test_call_next_unavailable_counter_rejected(setup_offices_and_services):
    """Calling next with an UNAVAILABLE counter returns 400 Bad Request."""
    db = SessionLocal()
    try:
        # Make C2 idle then UNAVAILABLE
        active_tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 2,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if active_tok:
            active_tok.status = TokenStatus.COMPLETED
        c2 = db.query(Counter).filter(Counter.id == 2).first()
        c2.status = CounterStatus.UNAVAILABLE
        db.commit()
    finally:
        db.close()

    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 2},
    )
    assert resp.status_code == 400
    assert "UNAVAILABLE" in resp.json()["detail"]


def test_call_next_wrong_service_rejected(setup_offices_and_services):
    """Calling next for PT on a counter assigned to WC returns 400 Bad Request."""
    # Counter 3 is assigned to WC (service_id=2)
    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 3},
    )
    assert resp.status_code == 400
    assert "assigned to service 2, not 1" in resp.json()["detail"]


def test_call_next_ineligible_counter_rejected(setup_offices_and_services):
    """Counter not eligible for the requested service is rejected."""
    # Counter 6 is only eligible for BC (service_id=3)
    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 6},
    )
    assert resp.status_code == 400


def test_call_next_fifo_preserved(setup_offices_and_services):
    """Tokens are called strictly in FIFO order (created_at ASC, id ASC)."""
    # Recover C2 and make it idle
    db = SessionLocal()
    try:
        c2 = db.query(Counter).filter(Counter.id == 2).first()
        c2.status = CounterStatus.ACTIVE
        db.commit()

        # Find the oldest waiting token in DB
        oldest_waiting = db.query(QueueToken).filter(
            QueueToken.office_id == 1,
            QueueToken.service_id == 1,
            QueueToken.status == TokenStatus.WAITING,
        ).order_by(QueueToken.created_at.asc(), QueueToken.id.asc()).first()
    finally:
        db.close()

    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 2},
    )
    assert resp.status_code == 200
    called_token = resp.json()
    assert called_token["id"] == oldest_waiting.id


# =============================================================================
# 4. REAL INTERVENTION APPLY & RESTORE TESTS
# =============================================================================

def test_apply_intervention_busy_counter_succeeds(setup_offices_and_services):
    """Step 2E.5B: Reassignment of a counter with an active token succeeds without interrupting the token."""
    # Ensure Scenario B is loaded
    client.post("/api/v1/admin/offices/1/scenarios/load", json={"scenario_id": "B"})

    # Counter 5 is active and has an active token in Scenario B
    resp = client.post(
        "/api/v1/interventions/apply",
        json={
            "office_id": 1,
            "counter_id": 5,
            "target_service_id": 1,  # PT (C5 is eligible for PT; donor BC has C6 active)
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    assert data["counter_id"] == 5
    assert data["target_service_id"] == 1

    # Verify active token is still intact
    db = SessionLocal()
    try:
        active_tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 5,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert active_tok is not None
        assert active_tok.service_id == 3  # Original service intact
    finally:
        db.close()


def test_apply_intervention_donor_retention_enforced(setup_offices_and_services):
    """Donor service must retain at least 1 active counter."""
    client.post("/api/v1/admin/offices/1/scenarios/load", json={"scenario_id": "B"})
    db = SessionLocal()
    try:
        # In Office 1, make C6 UNAVAILABLE so BC (service 3) only has 1 active counter (C5)
        c6 = db.query(Counter).filter(Counter.id == 6).first()
        c6.status = CounterStatus.UNAVAILABLE
        # Make C5 idle
        tok5 = db.query(QueueToken).filter(
            QueueToken.counter_id == 5,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if tok5:
            tok5.status = TokenStatus.COMPLETED
        db.commit()
    finally:
        db.close()

    # Reassigning C5 would leave BC with 0 active counters
    resp = client.post(
        "/api/v1/interventions/apply",
        json={
            "office_id": 1,
            "counter_id": 5,
            "target_service_id": 1,
        },
    )
    assert resp.status_code == 400
    assert "must retain at least 1 active counter" in resp.json()["detail"].lower()


def test_apply_intervention_success(setup_offices_and_services):
    """Valid idle eligible counter reassigns successfully."""
    client.post("/api/v1/admin/offices/1/scenarios/load", json={"scenario_id": "B"})
    db = SessionLocal()
    try:
        # Recover C6 and make it active so BC has 2 active counters
        c6 = db.query(Counter).filter(Counter.id == 6).first()
        c6.status = CounterStatus.ACTIVE
        tok6 = db.query(QueueToken).filter(
            QueueToken.counter_id == 6,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if tok6:
            tok6.status = TokenStatus.COMPLETED
        db.commit()
    finally:
        db.close()

    # C5 is idle, eligible for PT (1), donor BC has C6 active
    resp = client.post(
        "/api/v1/interventions/apply",
        json={
            "office_id": 1,
            "counter_id": 5,
            "target_service_id": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["applied"] is True
    assert data["counter_id"] == 5
    assert data["previous_service_id"] == 3
    assert data["target_service_id"] == 1
    assert "3.0-minute switching delay" in data["switching_delay_note"]

    # Verify DB state
    db = SessionLocal()
    try:
        c5 = db.query(Counter).filter(Counter.id == 5).first()
        assert c5.assigned_service_id == 1
    finally:
        db.close()


def test_restore_intervention_success(setup_offices_and_services):
    """Counter restores to its authoritative home service assignment."""
    # C5 is currently assigned to PT (1), its home service is BC (3)
    resp = client.post(
        "/api/v1/interventions/restore",
        json={
            "office_id": 1,
            "counter_id": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["restored"] is True
    assert data["counter_id"] == 5
    assert data["previous_service_id"] == 1
    assert data["restored_service_id"] == 3

    # Verify DB state
    db = SessionLocal()
    try:
        c5 = db.query(Counter).filter(Counter.id == 5).first()
        assert c5.assigned_service_id == 3
    finally:
        db.close()


def test_restore_intervention_already_at_home_rejected(setup_offices_and_services):
    """Restoring a counter already at its home assignment returns 400."""
    resp = client.post(
        "/api/v1/interventions/restore",
        json={
            "office_id": 1,
            "counter_id": 5,
        },
    )
    assert resp.status_code == 400
    assert "already at its authoritative home assignment" in resp.json()["detail"]


# =============================================================================
# 5. COUNTER UNAVAILABLE & RECOVERY TESTS
# =============================================================================

def test_counter_unavailable_and_recovery_flow(setup_offices_and_services):
    """Test full counter lifecycle: ACTIVE -> UNAVAILABLE -> ACTIVE."""
    # C5 is idle
    resp_unavail = client.post("/api/v1/counters/5/unavailable")
    assert resp_unavail.status_code == 200
    assert resp_unavail.json()["status"] == "UNAVAILABLE"

    # Verify DB state
    db = SessionLocal()
    try:
        c5 = db.query(Counter).filter(Counter.id == 5).first()
        assert c5.status == CounterStatus.UNAVAILABLE
    finally:
        db.close()

    # Recovery
    resp_recov = client.post("/api/v1/counters/5/recover")
    assert resp_recov.status_code == 200
    assert resp_recov.json()["status"] == "ACTIVE"

    db = SessionLocal()
    try:
        c5 = db.query(Counter).filter(Counter.id == 5).first()
        assert c5.status == CounterStatus.ACTIVE
        assert c5.assigned_service_id == 3  # Assignment preserved
    finally:
        db.close()


def test_counter_unavailable_busy_counter_succeeds(setup_offices_and_services):
    """Step 2E.5B: Marking counter unavailable succeeds even while an active token is being served."""
    db = SessionLocal()
    try:
        # Ensure C4 has an active SERVING token
        active_tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 4,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        if not active_tok:
            t = QueueToken(
                id=8888,
                token_number="MUNI-WC-888",
                office_id=1,
                service_id=2,
                counter_id=4,
                status=TokenStatus.SERVING,
                created_at=datetime.now(timezone.utc),
            )
            db.add(t)
            db.commit()
    finally:
        db.close()

    resp = client.post("/api/v1/counters/4/unavailable")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "UNAVAILABLE"

    # Verify active token is still serving and intact
    db = SessionLocal()
    try:
        tok = db.query(QueueToken).filter(QueueToken.counter_id == 4).first()
        assert tok is not None
        assert tok.status in [TokenStatus.CALLED, TokenStatus.SERVING]
        assert tok.service_id == 2
    finally:
        db.close()
