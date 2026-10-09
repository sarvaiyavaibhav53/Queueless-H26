"""
Tests for deterministic arrival rate calculation, window boundaries, eligible token statuses,
units, traffic intensity, time-to-congestion (TTC) fluid approximation, and canonical ML feature alignment.
"""

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from app.api.admin import compute_time_to_congestion_fluid
from app.db.database import SessionLocal
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.services.snapshot_adapter import build_simulation_snapshot
from shared.features import (
    compute_raw_features,
    SimulationSnapshot,
    ServiceQueueSnapshot,
    ServiceConfigSnapshot,
    CounterSnapshot,
    ServingTokenSnapshot,
    QueueCitizenSnapshot,
)


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_time_to_congestion_fluid_formula_units_and_boundaries():
    """Verify units, fluid accumulation, boundary conditions, and edge cases."""
    # Scenario B PT: L_q = 8, C = 15, lambda = 13 citizens/hr, mu = 12 citizens/hr
    # Net growth = 1 citizen/hr = 1/60 citizens/min.
    # Remaining = 7 tokens.
    # TTC = 7 / (1/60) = 420.0 minutes.
    ttc, method = compute_time_to_congestion_fluid(
        queue_length=8,
        arrival_rate_per_hour=13.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert method == "fluid_approximation"
    assert ttc == 420.0

    # Old rehearsal discrepancy scenario: lambda = 23 citizens/hr, mu = 12 citizens/hr
    # Net growth = 11 citizens/hr = 11/60 citizens/min.
    # Remaining = 7 tokens.
    # TTC = 7 / (11/60) = 38.18... -> 38.2 minutes.
    ttc_old, _ = compute_time_to_congestion_fluid(
        queue_length=8,
        arrival_rate_per_hour=23.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert ttc_old == 38.2

    # Already congested: queue_length >= congestion_threshold
    ttc_cong, _ = compute_time_to_congestion_fluid(
        queue_length=15,
        arrival_rate_per_hour=20.0,
        service_capacity_per_hour=10.0,
        congestion_threshold=15,
    )
    assert ttc_cong == 0.0

    # Stable or draining: lambda <= mu
    ttc_stable, _ = compute_time_to_congestion_fluid(
        queue_length=8,
        arrival_rate_per_hour=12.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert ttc_stable is None

    ttc_draining, _ = compute_time_to_congestion_fluid(
        queue_length=8,
        arrival_rate_per_hour=8.0,
        service_capacity_per_hour=12.0,
        congestion_threshold=15,
    )
    assert ttc_draining is None

    # Invalid / negative values
    ttc_neg, _ = compute_time_to_congestion_fluid(-1, 10.0, 5.0, 15)
    assert ttc_neg is None

    ttc_bad_thresh, _ = compute_time_to_congestion_fluid(5, 10.0, 5.0, 0)
    assert ttc_bad_thresh is None


def test_arrival_window_boundaries_and_token_eligibility(db_session: Session):
    """
    Verify snapshot adapter arrival window boundaries:
      - (t - 60m, t] for 60m window
      - (t - 30m, t] for 30m window
      - Eligible event types: WAITING, SERVING, COMPLETED, CANCELLED
    """
    ref_time = datetime(2026, 3, 30, 11, 0, 0, tzinfo=timezone.utc)
    office_id = 999
    service_id = 999
    counter_id = 999

    # Clean up test IDs
    db_session.query(QueueToken).filter(QueueToken.office_id == office_id).delete()
    db_session.query(CounterService).filter(CounterService.counter_id == counter_id).delete()
    db_session.query(Counter).filter(Counter.id == counter_id).delete()
    db_session.query(Service).filter(Service.id == service_id).delete()
    db_session.query(Office).filter(Office.id == office_id).delete()
    db_session.commit()

    try:
        # Create Office and Service
        office = Office(id=office_id, code="TST", name="Test Window Office", counters=1, demand_multiplier=1.0)
        service = Service(id=service_id, code="TS", name="Test Service", mean_minutes=10.0, std_minutes=2.0)
        db_session.add(office)
        db_session.add(service)
        db_session.flush()

        # Create Counter
        counter = Counter(
            id=counter_id,
            office_id=office_id,
            name="C-999",
            status=CounterStatus.ACTIVE,
            assigned_service_id=service_id,
            status_changed_at=ref_time - timedelta(hours=2),
        )
        db_session.add(counter)
        db_session.flush()
        db_session.add(CounterService(counter_id=counter_id, service_id=service_id))
        db_session.flush()

        # Seed tokens with specific arrival timestamps
        # Window definition: t_snap = 11:00:00
        # t - 60m = 10:00:00. Bound: (10:00:00, 11:00:00]
        # t - 30m = 10:30:00. Bound: (10:30:00, 11:00:00]

        tokens_data = [
            # 1. 61m ago: 09:59:00 -> OUTSIDE 60m window
            (9901, TokenStatus.COMPLETED, ref_time - timedelta(minutes=61)),
            # 2. Exactly 60m ago: 10:00:00 -> OUTSIDE (strict > lookback)
            (9902, TokenStatus.COMPLETED, ref_time - timedelta(minutes=60)),
            # 3. 55m ago: 10:05:00 -> INSIDE 60m, OUTSIDE 30m (COMPLETED)
            (9903, TokenStatus.COMPLETED, ref_time - timedelta(minutes=55)),
            # 4. 45m ago: 10:15:00 -> INSIDE 60m, OUTSIDE 30m (CANCELLED)
            (9904, TokenStatus.CANCELLED, ref_time - timedelta(minutes=45)),
            # 5. Exactly 30m ago: 10:30:00 -> INSIDE 60m, OUTSIDE 30m (strict > t_30)
            (9905, TokenStatus.COMPLETED, ref_time - timedelta(minutes=30)),
            # 6. 25m ago: 10:35:00 -> INSIDE 60m, INSIDE 30m (SERVING at counter)
            (9906, TokenStatus.SERVING, ref_time - timedelta(minutes=25)),
            # 7. 15m ago: 10:45:00 -> INSIDE 60m, INSIDE 30m (WAITING)
            (9907, TokenStatus.WAITING, ref_time - timedelta(minutes=15)),
            # 8. 5m ago: 10:55:00 -> INSIDE 60m, INSIDE 30m (WAITING)
            (9908, TokenStatus.WAITING, ref_time - timedelta(minutes=5)),
            # 9. Exactly ref_time: 11:00:00 -> INSIDE 60m, INSIDE 30m (WAITING)
            (9909, TokenStatus.WAITING, ref_time),
            # 10. Future token: 11:05:00 -> OUTSIDE both (created_at > snapshot_time)
            (9910, TokenStatus.WAITING, ref_time + timedelta(minutes=5)),
        ]

        for t_id, t_stat, t_arr in tokens_data:
            tok = QueueToken(
                id=t_id,
                token_number=f"TST-{t_id}",
                office_id=office_id,
                service_id=service_id,
                status=t_stat,
                created_at=t_arr,
                counter_id=counter_id if t_stat == TokenStatus.SERVING else None,
                called_at=t_arr + timedelta(minutes=1) if t_stat in (TokenStatus.SERVING, TokenStatus.COMPLETED) else None,
                serving_at=t_arr + timedelta(minutes=2) if t_stat in (TokenStatus.SERVING, TokenStatus.COMPLETED) else None,
            )
            db_session.add(tok)
        db_session.commit()

        # Build Simulation Snapshot
        snap = build_simulation_snapshot(
            db=db_session,
            office_id=office_id,
            snapshot_time_utc=ref_time,
        )

        # Expected inside 60m window:
        # 9903 (55m), 9904 (45m), 9905 (30m), 9906 (25m), 9907 (15m), 9908 (5m), 9909 (0m) -> 7 arrivals
        q_snap = snap.queues[service_id]
        assert len(q_snap.recent_arrivals_utc) == 7

        # Extract canonical ML features for a waiting token at position 3
        features = compute_raw_features(
            snapshot=snap,
            service_id=service_id,
            position=3,
        )

        # In shared/features:
        # count_30: t_30 < t_arr <= t_snap
        # Inside 30m: 9906 (25m), 9907 (15m), 9908 (5m), 9909 (0m) -> 4 arrivals
        # arrival_rate_30m = 4 / 0.5 = 8.0 citizens/hr
        # count_60: t_60 < t_arr <= t_snap
        # Inside 60m: 7 arrivals -> arrival_rate_60m = 7 / 1.0 = 7.0 citizens/hr
        assert features.arrival_rate_30m == 8.0
        assert features.arrival_rate_60m == 7.0

        # Service capacity: 1 counter * (60 / 10.0) = 6.0 citizens/hr
        assert features.service_capacity_mu == 6.0

        # Traffic intensity: 7.0 / 6.0 = 1.166666...
        assert abs(features.traffic_intensity - (7.0 / 6.0)) < 1e-4

        # Verify waiting queue length: 9907, 9908, 9909 -> 3 waiting tokens
        assert features.queue_length == 3
        # People ahead for position 3: 2
        assert features.people_ahead == 2

    finally:
        db_session.query(QueueToken).filter(QueueToken.office_id == office_id).delete()
        db_session.query(CounterService).filter(CounterService.counter_id == counter_id).delete()
        db_session.query(Counter).filter(Counter.id == counter_id).delete()
        db_session.query(Service).filter(Service.id == service_id).delete()
        db_session.query(Office).filter(Office.id == office_id).delete()
        db_session.commit()
