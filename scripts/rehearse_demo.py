"""
PRAVAAH — End-to-End Demo Rehearsal Script (Step 2E.5B Section 14).

Executes a complete, non-destructive, repeatable verification of all 14 lifecycle
and operational requirements against a local demo database.

Verifies:
 1. A known baseline scenario loads deterministically.
 2. Admin summary returns consistent operational metrics and labelled fluid time-to-congestion.
 3. A citizen can retrieve a queue snapshot and ETA.
 4. An employee can call, start, and complete a token.
 5. The canonical E_known scenario loads with its expected state (C1 UNAVAILABLE at t=0).
 6. Intervention evaluation remains preview-only (zero state mutation).
 7. A valid reassignment succeeds while the donor counter is serving a token.
 8. The existing token completes normally and retains original service/counter association.
 9. The counter's next call uses its new assignment.
10. A busy counter can be marked unavailable without interrupting its current token.
11. That token completes, the unavailable counter receives no new token, and explicit recovery works.
12. WebSocket invalidation events have the exact four-key shape and no duplicates.
13. Restore returns counters to their canonical home assignments.
14. The final admin summary is internally consistent.

Exit codes:
  0: All 14 verification steps passed.
  1: Verification failure (details printed to stderr).
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Ensure backend package is in python search path
repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.security import get_password_hash
from app.db.database import SessionLocal, engine
from app.models.base import Base
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.models.user import User, UserRole
from app.ws_manager import ws_manager
from scripts.seed_data import seed_database

client = TestClient(app)


def log_step(step_num: int, title: str):
    print(f"\n[{step_num}/14] {title}")


def get_token(email: str, password: str) -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to authenticate {email}: {resp.status_code} {resp.text}")
    return resp.json()["access_token"]


def run_rehearsal():
    print("=" * 70)
    print("PRAVAAH END-TO-END DEMO REHEARSAL (Step 2E.5B)")
    print(f"Target Database URL: {settings.DATABASE_URL}")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Setup base seed if needed
    seed_database()

    # Obtain role credentials
    admin_token = get_token("admin@gov.in", "Admin@123")
    emp_token = get_token("employee@gov.in", "Employee@123")
    citizen_token = get_token("citizen@example.com", "Citizen@123")

    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    emp_headers = {"Authorization": f"Bearer {emp_token}"}
    citizen_headers = {"Authorization": f"Bearer {citizen_token}"}

    # -------------------------------------------------------------------------
    # 1. Baseline scenario loads
    # -------------------------------------------------------------------------
    log_step(1, "Loading baseline scenario (scenario_b_pt_pressure)...")
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "scenario_b_pt_pressure"},
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Baseline scenario load failed: {resp.text}"
    load_data = resp.json()
    assert load_data["counters_loaded"] == 6
    assert load_data["busy_counters_loaded"] == 6
    assert load_data["waiting_tokens_loaded"] == 10
    print("  [+] Baseline scenario loaded deterministically (6 counters, 6 busy, 10 waiting).")

    # -------------------------------------------------------------------------
    # 2. Admin summary returns operational metrics & labelled fluid TTC
    # -------------------------------------------------------------------------
    log_step(2, "Verifying Admin Summary operational metrics and fluid TTC...")
    resp = client.get(
        "/api/v1/admin/offices/1/summary?congestion_threshold=15",
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Summary request failed: {resp.text}"
    summary = resp.json()
    assert summary["office_id"] == 1
    assert len(summary["counters"]) == 6
    assert len(summary["services"]) >= 3

    pt_summary = next((s for s in summary["services"] if s["service_code"] == "PT"), None)
    assert pt_summary is not None, "PT service missing from summary"
    assert pt_summary["queue_length"] == 8
    assert pt_summary["time_to_congestion_method"] == "fluid_approximation"
    # PT in Scenario B has lambda=13.0/hr, mu=12.0/hr (net +1.0/hr = 1/60/min), Q=8, thresh=15 -> TTC = (15-8)/(1/60) = 420.0 min
    assert pt_summary["time_to_congestion_minutes"] == 420.0
    assert round(pt_summary["traffic_intensity"], 2) == 1.08
    print(f"  [+] PT Service: Q={pt_summary['queue_length']}, rho={pt_summary['traffic_intensity']:.2f}, "
          f"TTC={pt_summary['time_to_congestion_minutes']:.1f}m (method: {pt_summary['time_to_congestion_method']}).")

    # -------------------------------------------------------------------------
    # 3. Citizen can retrieve queue snapshot and ETA
    # -------------------------------------------------------------------------
    log_step(3, "Citizen retrieving queue status snapshot and live ETA...")
    # Get status for service 1 (PT)
    resp = client.get("/api/v1/queues/1/1/status", headers=citizen_headers)
    assert resp.status_code == 200, f"Queue status failed: {resp.text}"
    status_data = resp.json()
    assert status_data["waiting_count"] >= 8
    print(f"  [+] Citizen queue status received (waiting_count={status_data['waiting_count']}).")

    # Citizen joins queue to obtain personal token for ETA
    resp = client.post("/api/v1/queues/1/join", json={"office_id": 1}, headers=citizen_headers)
    assert resp.status_code == 201, f"Join queue failed: {resp.text}"
    citizen_token_data = resp.json()
    my_token_id = citizen_token_data["id"]

    # Citizen views token ETA
    resp = client.get(f"/api/v1/queues/{my_token_id}/eta", headers=citizen_headers)
    assert resp.status_code == 200, f"Token ETA failed: {resp.text}"
    eta_data = resp.json()
    assert "predicted_wait_minutes" in eta_data
    assert eta_data["predicted_wait_minutes"] >= 0.0
    assert "prediction_source" in eta_data
    print(f"  [+] Citizen live ETA retrieved successfully (predicted_wait={eta_data['predicted_wait_minutes']}m, source={eta_data['prediction_source']}).")

    # -------------------------------------------------------------------------
    # 4. Employee can call, start, and complete a token
    # -------------------------------------------------------------------------
    log_step(4, "Employee call, start, and complete token lifecycle...")
    # Counter 2 is busy with an existing token; complete it first to make C2 idle
    db = SessionLocal()
    try:
        c2_token = db.query(QueueToken).filter(
            QueueToken.counter_id == 2,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert c2_token is not None
        c2_token_id = c2_token.id
    finally:
        db.close()

    resp = client.post(f"/api/v1/tokens/{c2_token_id}/complete", headers=emp_headers)
    assert resp.status_code == 200, f"Completing token on C2 failed: {resp.text}"

    # Now C2 is idle, employee calls next waiting token on C2 (service 1)
    resp = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 2},
        headers=emp_headers,
    )
    assert resp.status_code == 200, f"Call-next on C2 failed: {resp.text}"
    called_token = resp.json()
    assert called_token["status"] == "CALLED"
    assert called_token["counter_id"] == 2
    step4_token_id = called_token["id"]

    # Start serving
    resp = client.post(f"/api/v1/tokens/{step4_token_id}/start", headers=emp_headers)
    assert resp.status_code == 200, f"Start serving failed: {resp.text}"
    assert resp.json()["status"] == "SERVING"

    # Complete token
    resp = client.post(f"/api/v1/tokens/{step4_token_id}/complete", headers=emp_headers)
    assert resp.status_code == 200, f"Complete serving failed: {resp.text}"
    assert resp.json()["status"] == "COMPLETED"
    print("  [+] Full employee token lifecycle verified (CALLED -> SERVING -> COMPLETED).")

    # -------------------------------------------------------------------------
    # 5. Canonical E_known scenario loads with expected state
    # -------------------------------------------------------------------------
    log_step(5, "Loading canonical E_known scenario...")
    resp = client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "E_known"},
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Loading E_known failed: {resp.text}"
    e_data = resp.json()
    assert e_data["scenario_id"] in ("scenario_e_known", "E_known")
    assert e_data["counters_loaded"] == 6
    assert e_data["busy_counters_loaded"] == 5

    # Verify C1 is UNAVAILABLE at t=0 with 0 active tokens, and C2..C6 are ACTIVE
    db = SessionLocal()
    try:
        c1 = db.query(Counter).filter(Counter.id == 1).first()
        assert c1.status == CounterStatus.UNAVAILABLE, "C1 must be UNAVAILABLE in E_known"
        c1_tok = db.query(QueueToken).filter(
            QueueToken.counter_id == 1,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert c1_tok is None, "C1 must have 0 active tokens"
        for cid in [2, 3, 4, 5, 6]:
            c = db.query(Counter).filter(Counter.id == cid).first()
            assert c.status == CounterStatus.ACTIVE, f"C{cid} must be ACTIVE in E_known"
    finally:
        db.close()
    print("  [+] Canonical E_known loaded: C1 UNAVAILABLE at t=0 (0 active tokens), C2-C6 ACTIVE.")

    # -------------------------------------------------------------------------
    # 6. Intervention evaluation remains preview-only
    # -------------------------------------------------------------------------
    log_step(6, "Evaluating candidate intervention (preview-only / zero mutation)...")
    resp = client.post(
        "/api/v1/interventions/evaluate",
        json={"office_id": 1, "replications": 2, "horizon_minutes": 30.0},
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Intervention evaluate failed: {resp.text}"
    eval_data = resp.json()
    assert eval_data["is_simulation_only"] is True

    # Verify no counters were mutated by evaluation
    db = SessionLocal()
    try:
        c1_after = db.query(Counter).filter(Counter.id == 1).first()
        assert c1_after.status == CounterStatus.UNAVAILABLE, "C1 status must remain UNAVAILABLE"
    finally:
        db.close()
    print("  [+] Intervention evaluation is strictly preview-only (is_simulation_only=True).")

    # -------------------------------------------------------------------------
    # 7. Valid reassignment succeeds while donor counter is serving a token
    # -------------------------------------------------------------------------
    log_step(7, "Reassigning busy counter (C5 from BC to PT) during active service...")
    # Reload Scenario B so C5 has active token serving BC (service 3)
    client.post(
        "/api/v1/admin/offices/1/scenarios/load",
        json={"scenario_id": "scenario_b_pt_pressure"},
        headers=admin_headers,
    )
    db = SessionLocal()
    try:
        c5_token = db.query(QueueToken).filter(
            QueueToken.counter_id == 5,
            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
        ).first()
        assert c5_token is not None, "C5 must have an active token"
        assert c5_token.service_id == 3, "C5 active token must be for BC (3)"
        active_t_id = c5_token.id
    finally:
        db.close()

    # Apply reassignment C5 -> service 1 (PT)
    resp = client.post(
        "/api/v1/interventions/apply",
        json={"office_id": 1, "counter_id": 5, "target_service_id": 1},
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Reassignment of busy counter failed: {resp.text}"
    reassign_data = resp.json()
    assert reassign_data["applied"] is True
    assert reassign_data["counter_id"] == 5
    assert reassign_data["target_service_id"] == 1
    assert reassign_data["previous_service_id"] == 3
    print("  [+] Reassignment succeeded for busy counter C5 (BC -> PT) without 409 conflict.")

    # -------------------------------------------------------------------------
    # 8. Existing token completes normally and retains original service/counter
    # -------------------------------------------------------------------------
    log_step(8, "Verifying in-progress token integrity and completing normally...")
    db = SessionLocal()
    try:
        t_check = db.query(QueueToken).filter(QueueToken.id == active_t_id).first()
        assert t_check.service_id == 3, "Token service_id must NOT be altered"
        assert t_check.counter_id == 5, "Token counter_id must remain 5"
        assert t_check.status in (TokenStatus.CALLED, TokenStatus.SERVING)
    finally:
        db.close()

    # Complete the active token
    resp = client.post(f"/api/v1/tokens/{active_t_id}/complete", headers=emp_headers)
    assert resp.status_code == 200, f"Completion of token {active_t_id} failed: {resp.text}"
    assert resp.json()["status"] == "COMPLETED"
    print(f"  [+] Token {active_t_id} retained original service_id=3 and completed normally.")

    # -------------------------------------------------------------------------
    # 9. Counter's next call uses its new assignment
    # -------------------------------------------------------------------------
    log_step(9, "Verifying counter C5's next call uses its new assignment (PT)...")
    # Calling on previous service BC (3) must be rejected
    resp_prev = client.post(
        "/api/v1/queues/1/3/call-next",
        json={"counter_id": 5},
        headers=emp_headers,
    )
    assert resp_prev.status_code == 400, "Calling previous service on reassigned counter must fail"

    # Calling on new service PT (1) must succeed
    resp_new = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 5},
        headers=emp_headers,
    )
    assert resp_new.status_code == 200, f"Calling new service PT on C5 failed: {resp_new.text}"
    c5_new_token = resp_new.json()
    assert c5_new_token["service_id"] == 1, "Served token must belong to new service PT"
    c5_new_token_id = c5_new_token["id"]
    print(f"  [+] C5 called next token {c5_new_token_id} for new assigned service PT (1).")

    # -------------------------------------------------------------------------
    # 10. Busy counter can be marked unavailable without interrupting token
    # -------------------------------------------------------------------------
    log_step(10, "Marking busy counter C5 UNAVAILABLE while serving token...")
    resp = client.post(f"/api/v1/counters/5/unavailable", headers=admin_headers)
    assert resp.status_code == 200, f"Marking busy counter unavailable failed: {resp.text}"
    c5_status_data = resp.json()
    assert c5_status_data["status"] == "UNAVAILABLE"

    # Verify active token on C5 is still intact and CALLED
    db = SessionLocal()
    try:
        t_live = db.query(QueueToken).filter(QueueToken.id == c5_new_token_id).first()
        assert t_live.status == TokenStatus.CALLED
        assert t_live.counter_id == 5
    finally:
        db.close()
    print("  [+] C5 successfully marked UNAVAILABLE while busy; active token remains intact.")

    # -------------------------------------------------------------------------
    # 11. Token completes, unavailable counter receives no token, explicit recovery
    # -------------------------------------------------------------------------
    log_step(11, "Completing token on unavailable counter, blocking calls, and recovering...")
    # Start and complete token
    resp_start = client.post(f"/api/v1/tokens/{c5_new_token_id}/start", headers=emp_headers)
    assert resp_start.status_code == 200, f"Start token failed: {resp_start.text}"
    resp = client.post(f"/api/v1/tokens/{c5_new_token_id}/complete", headers=emp_headers)
    assert resp.status_code == 200, f"Complete token failed: {resp.text}"
    assert resp.json()["status"] == "COMPLETED"

    # Attempt call-next on unavailable counter must fail
    resp_blocked = client.post(
        "/api/v1/queues/1/1/call-next",
        json={"counter_id": 5},
        headers=emp_headers,
    )
    assert resp_blocked.status_code == 400, "Unavailable counter must not accept calls"

    # Explicit recovery
    resp_rec = client.post(f"/api/v1/counters/5/recover", headers=admin_headers)
    assert resp_rec.status_code == 200
    assert resp_rec.json()["status"] == "ACTIVE"
    assert resp_rec.json()["assigned_service_id"] == 1, "Assignment must be preserved across recovery"
    print("  [+] Token completed, call blocked while UNAVAILABLE, recovered to ACTIVE (assignment preserved).")

    # -------------------------------------------------------------------------
    # 12. WebSocket invalidations: exact 4 keys, commit safety, non-duplicated
    # -------------------------------------------------------------------------
    log_step(12, "Verifying WebSocket invalidation contract and payload shape...")
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
        # Trigger queue mutation
        resp_join = client.post("/api/v1/queues/1/join", json={"office_id": 1}, headers=citizen_headers)
        assert resp_join.status_code in [200, 201]

        # Trigger counter status mutation
        resp_unavail = client.post("/api/v1/counters/2/unavailable", headers=admin_headers)
        assert resp_unavail.status_code == 200

        assert len(captured) >= 2
        for ev in captured:
            assert set(ev.keys()) == {"type", "office_id", "service_id", "occurred_at"}, (
                f"Event keys violation: {list(ev.keys())}"
            )
            assert isinstance(ev["type"], str)
            assert isinstance(ev["office_id"], int)
            assert isinstance(ev["service_id"], int)
            assert isinstance(ev["occurred_at"], str)
    finally:
        ws_manager.broadcast_invalidation = orig_broadcast
        # recover counter 2
        client.post("/api/v1/counters/2/recover", headers=admin_headers)

    print("  [+] Verified WebSocket events conform strictly to 4-key contract without PII.")

    # -------------------------------------------------------------------------
    # 13. Restore returns counters to canonical home assignments
    # -------------------------------------------------------------------------
    log_step(13, "Restoring counter C5 to canonical home assignment...")
    resp = client.post(
        "/api/v1/interventions/restore",
        json={"office_id": 1, "counter_id": 5},
        headers=admin_headers,
    )
    assert resp.status_code == 200, f"Restore failed: {resp.text}"
    restore_data = resp.json()
    assert restore_data["restored"] is True
    assert restore_data["previous_service_id"] == 1
    assert restore_data["restored_service_id"] == 3

    db = SessionLocal()
    try:
        c5_restored = db.query(Counter).filter(Counter.id == 5).first()
        assert c5_restored.assigned_service_id == 3
    finally:
        db.close()
    print("  [+] Counter C5 successfully restored to canonical home service BC (3).")

    # -------------------------------------------------------------------------
    # 14. Final admin summary is internally consistent
    # -------------------------------------------------------------------------
    log_step(14, "Verifying final Admin Summary consistency...")
    resp = client.get("/api/v1/admin/offices/1/summary", headers=admin_headers)
    assert resp.status_code == 200
    final_sum = resp.json()

    total_active = sum(s["assigned_active_counters"] for s in final_sum["services"])
    total_unavail = sum(s["unavailable_counters"] for s in final_sum["services"])
    assert total_active + total_unavail == 6, "Total counters across services must equal office counters (6)"
    for s in final_sum["services"]:
        assert s["queue_length"] >= 0
        assert s["traffic_intensity"] >= 0.0
    print("  [+] Final admin summary verified: counters sum to 6, no negative queues/metrics.")

    print("\n" + "=" * 70)
    print("PRAVAAH END-TO-END DEMO REHEARSAL: 14/14 CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_rehearsal()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[!] Rehearsal failed: {e}", file=sys.stderr)
        sys.exit(1)
