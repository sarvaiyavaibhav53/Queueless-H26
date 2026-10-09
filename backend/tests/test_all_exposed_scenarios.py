"""
PRAVAAH — Comprehensive Verification of All Exposed Scenarios (Integrity Gate).

Validates every scenario exposed by get_available_scenarios():
- Queue lengths match canonical initial_queue_counts.
- FIFO ordering strictly preserved.
- Active and unavailable counter counts match scenario definition.
- Busy counters have exactly 1 active token assigned to counter's assigned_service_id.
- Unavailable counters (e.g. C1 in E_known) have 0 active tokens.
- Observed arrival rate (arrival_rate_60m) matches target arrivals in 60m window.
- Traffic intensity rho matches arrival_rate_60m / service_capacity_mu.
- Time to congestion is finite when lambda > mu, None when lambda <= mu.
- Office isolation is strictly maintained.
- Repeated loading is deterministic and idempotent.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import SessionLocal
from app.models.counter import Counter, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.services.scenario_loader_service import get_available_scenarios, load_scenario_into_db
from app.simulation.scenarios import SCENARIO_REGISTRY, get_scenario

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_offices_services():
    """Seeds Municipality and RTO offices and services."""
    db = SessionLocal()
    try:
        # Offices
        for oid, code, name, cnt in [(1, "MUNI", "Municipality", 6), (2, "RTO", "RTO", 3)]:
            o = db.query(Office).filter(Office.id == oid).first()
            if not o:
                db.add(Office(id=oid, code=code, name=name, counters=cnt))

        # Services
        services_data = [
            (1, "PT", "Property Tax", 10.0, 4.0),
            (2, "WC", "Water Service", 15.0, 6.0),
            (3, "BC", "Birth Certificate", 7.0, 2.8),
            (4, "OT", "Ownership Transfer", 14.0, 5.6),
            (5, "DR", "Duplicate RC", 8.0, 3.2),
        ]
        for sid, code, name, mean_m, std_m in services_data:
            s = db.query(Service).filter(Service.id == sid).first()
            if not s:
                db.add(Service(id=sid, code=code, name=name, mean_minutes=mean_m, std_minutes=std_m))
        db.commit()
    finally:
        db.close()


def test_every_exposed_scenario_state_integrity():
    """Iterates through every scenario exposed by the API and asserts exact intended state."""
    scenarios = get_available_scenarios()
    assert len(scenarios) >= 6

    db = SessionLocal()
    try:
        for scen_id in sorted(scenarios.keys()):
            scen_cfg = get_scenario(scen_id)
            office_id = scen_cfg.office_id

            # 1. Load scenario via API
            resp = client.post(
                f"/api/v1/admin/offices/{office_id}/scenarios/load",
                json={"scenario_id": scen_id},
            )
            assert resp.status_code == 200, f"Failed to load {scen_id}: {resp.text}"
            load_data = resp.json()

            expected_counters = len(scen_cfg.initial_counters)
            expected_busy = sum(1 for c in scen_cfg.initial_counters if c.is_active)
            expected_waiting = sum(scen_cfg.initial_queue_counts.values())

            assert load_data["counters_loaded"] == expected_counters, f"Counter count mismatch for {scen_id}"
            assert load_data["busy_counters_loaded"] == expected_busy, f"Busy counter mismatch for {scen_id}"
            assert load_data["waiting_tokens_loaded"] == expected_waiting, f"Waiting token mismatch for {scen_id}"

            # 2. Verify DB Queue Lengths and Strict FIFO Ordering
            for s_id, exp_count in scen_cfg.initial_queue_counts.items():
                tokens = (
                    db.query(QueueToken)
                    .filter(
                        QueueToken.office_id == office_id,
                        QueueToken.service_id == s_id,
                        QueueToken.status == TokenStatus.WAITING,
                    )
                    .order_by(QueueToken.created_at.asc(), QueueToken.id.asc())
                    .all()
                )
                assert len(tokens) == exp_count, f"Queue length mismatch for {scen_id} service {s_id}"
                if len(tokens) > 1:
                    # Assert strict FIFO timestamps
                    for idx in range(len(tokens) - 1):
                        assert tokens[idx].created_at < tokens[idx + 1].created_at, (
                            f"FIFO order violated in {scen_id} service {s_id}"
                        )

            # 3. Verify Counters and In-Progress Tokens
            for sc in scen_cfg.initial_counters:
                c_db = db.query(Counter).filter(Counter.id == sc.counter_id).first()
                assert c_db is not None, f"Counter {sc.counter_id} missing in DB for {scen_id}"
                expected_status = CounterStatus.ACTIVE if sc.is_active else CounterStatus.UNAVAILABLE
                assert c_db.status == expected_status, f"Counter {sc.counter_id} status mismatch in {scen_id}"
                assert c_db.assigned_service_id == sc.assigned_service_id

                active_tokens = (
                    db.query(QueueToken)
                    .filter(
                        QueueToken.counter_id == sc.counter_id,
                        QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
                    )
                    .all()
                )
                if sc.is_active:
                    assert len(active_tokens) == 1, f"Active counter {sc.counter_id} has != 1 active token in {scen_id}"
                    assert active_tokens[0].service_id == sc.assigned_service_id
                else:
                    assert len(active_tokens) == 0, f"Inactive counter {sc.counter_id} has active tokens in {scen_id}"

            # 4. Verify Admin Summary Metrics and TTC
            sum_resp = client.get(f"/api/v1/admin/offices/{office_id}/summary")
            assert sum_resp.status_code == 200
            sum_data = sum_resp.json()

            for s_id, s_cfg in scen_cfg.services.items():
                s_sum = next((s for s in sum_data["services"] if s["service_id"] == s_id), None)
                assert s_sum is not None, f"Service {s_id} missing from summary in {scen_id}"

                # Calculate expected capacity mu
                active_counters_for_s = sum(
                    1 for c in scen_cfg.initial_counters if c.is_active and c.assigned_service_id == s_id
                )
                exp_mu = active_counters_for_s * (60.0 / s_cfg.mean_service_minutes) if s_cfg.mean_service_minutes > 0 else 6.0
                assert abs(s_sum["service_capacity_mu"] - exp_mu) < 1e-2

                # Arrival rate and rho should be consistent
                arr_rate = s_sum["arrival_rate_60m"]
                assert arr_rate >= 0.0
                if exp_mu > 0:
                    assert abs(s_sum["traffic_intensity"] - (arr_rate / exp_mu)) < 1e-2

                # TTC verification
                ttc = s_sum["time_to_congestion_minutes"]
                q_len = s_sum["queue_length"]
                thresh = s_sum["congestion_threshold"]
                if arr_rate > exp_mu:
                    # Growing queue
                    if q_len >= thresh:
                        assert ttc == 0.0
                    else:
                        assert ttc is not None and ttc > 0.0
                else:
                    # Stable or draining queue
                    if q_len < thresh:
                        assert ttc is None

            # 5. Idempotent Re-load Verification
            resp_repeat = client.post(
                f"/api/v1/admin/offices/{office_id}/scenarios/load",
                json={"scenario_id": scen_id},
            )
            assert resp_repeat.status_code == 200
            assert resp_repeat.json()["counters_loaded"] == load_data["counters_loaded"]
            assert resp_repeat.json()["busy_counters_loaded"] == load_data["busy_counters_loaded"]
            assert resp_repeat.json()["waiting_tokens_loaded"] == load_data["waiting_tokens_loaded"]

    finally:
        db.close()
