"""
PRAVAAH — Backend Simulation Snapshot Adapter.

Transforms live PostgreSQL database state into the pure domain SimulationSnapshot
defined in shared/features.py.

Guarantees & Constraints:
- Pure boundary layer: converts SQLAlchemy models -> plain frozen snapshot dataclasses.
- Read-only: does not mutate database records, acquire write locks, or change queue state.
- Zero system clock calls: requires explicit timezone-aware UTC snapshot_time_utc.
- No ML, prediction, simulator, or optimization logic.
- Fails loudly on corrupt database state (e.g. assigned_service not in counter_services).
- Strict FIFO ordering preserved for waiting tokens.
- Recent arrival timestamps derived strictly from actual historical QueueToken creation times.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session

from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
)


class SnapshotAdapterError(ValueError):
    """Raised when database state violates domain snapshot invariants or cannot be adapted."""
    pass


def _ensure_utc(dt: Optional[datetime], field_name: str = "datetime") -> Optional[datetime]:
    """
    Normalizes a datetime to timezone-aware UTC.
    If naive, interprets as UTC canonical timestamp.
    """
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        raise TypeError(f"{field_name} must be a datetime instance (got {type(dt)})")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_simulation_snapshot(
    db: Session,
    office_id: int,
    snapshot_time_utc: datetime,
) -> SimulationSnapshot:
    """
    Constructs a SimulationSnapshot for an office at a given snapshot_time_utc.

    Args:
        db: SQLAlchemy Session (used for read-only queries).
        office_id: Office ID to snapshot.
        snapshot_time_utc: Explicit point-in-time timestamp (must be timezone-aware UTC).

    Returns:
        SimulationSnapshot ready for feature computation or simulation.

    Consistency Note:
        This adapter performs read-only SELECT queries within the caller's session.
        For atomic point-in-time consistency across tables in high-concurrency environments,
        callers should run this within a REPEATABLE READ transaction.
    """
    # 1. Validate snapshot_time_utc explicitly
    if snapshot_time_utc is None:
        raise SnapshotAdapterError("snapshot_time_utc must be explicitly provided")
    if not isinstance(snapshot_time_utc, datetime):
        raise TypeError(f"snapshot_time_utc must be a datetime (got {type(snapshot_time_utc)})")
    if snapshot_time_utc.tzinfo is None or snapshot_time_utc.utcoffset() != timedelta(0):
        raise SnapshotAdapterError(
            f"snapshot_time_utc must be a timezone-aware UTC datetime (got {snapshot_time_utc})"
        )

    if office_id <= 0:
        raise SnapshotAdapterError(f"office_id must be a positive integer (got {office_id})")

    # 2. Fetch Office
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise SnapshotAdapterError(f"Office with id {office_id} not found in database")

    # 3. Fetch Service configurations
    # We fetch all configured services so any counter assignment or queue lookup is valid
    db_services = db.query(Service).all()
    if not db_services:
        raise SnapshotAdapterError("No services configured in database")

    services_dict: Dict[int, ServiceConfigSnapshot] = {}
    for s in db_services:
        if s.mean_minutes is None or s.mean_minutes <= 0.0:
            raise SnapshotAdapterError(
                f"Service {s.id} ({s.code}) has invalid mean_minutes: {s.mean_minutes}"
            )
        std_val = float(s.std_minutes) if s.std_minutes is not None else 0.0
        services_dict[s.id] = ServiceConfigSnapshot(
            service_id=s.id,
            service_code=s.code,
            service_name=s.name,
            mean_service_minutes=float(s.mean_minutes),
            std_service_minutes=std_val,
        )

    # 4. Fetch Physical Counters for this Office
    db_counters = (
        db.query(Counter)
        .filter(Counter.office_id == office_id)
        .order_by(Counter.id.asc())
        .all()
    )

    counters_dict: Dict[int, CounterSnapshot] = {}
    active_token_ids_across_counters: Set[int] = set()

    for counter in db_counters:
        # Resolve eligible service IDs from counter_services
        eligible_query = (
            db.query(CounterService.service_id)
            .filter(CounterService.counter_id == counter.id)
            .all()
        )
        eligible_service_ids = frozenset(row[0] for row in eligible_query)

        # Invariant: Fail loudly if no eligibility or assigned service is ineligible
        if not eligible_service_ids:
            raise SnapshotAdapterError(
                f"Counter {counter.id} ('{counter.name}') has no eligible services configured in counter_services"
            )
        if counter.assigned_service_id not in eligible_service_ids:
            raise SnapshotAdapterError(
                f"Counter {counter.id} ('{counter.name}') has assigned_service_id "
                f"{counter.assigned_service_id} which is NOT in its eligible services {set(eligible_service_ids)}"
            )
        if counter.assigned_service_id not in services_dict:
            raise SnapshotAdapterError(
                f"Counter {counter.id} ('{counter.name}') assigned_service_id "
                f"{counter.assigned_service_id} is not a valid registered service"
            )

        status_str = counter.status.value if hasattr(counter.status, "value") else str(counter.status)
        if status_str not in ("ACTIVE", "UNAVAILABLE"):
            raise SnapshotAdapterError(
                f"Counter {counter.id} has invalid status '{status_str}' (expected ACTIVE or UNAVAILABLE)"
            )

        status_changed_at = _ensure_utc(counter.status_changed_at, "status_changed_at")
        if status_changed_at is None:
            raise SnapshotAdapterError(f"Counter {counter.id} has null status_changed_at")

        # Find active (CALLED or SERVING) token engaged at this counter desk
        active_tokens = (
            db.query(QueueToken)
            .filter(
                QueueToken.counter_id == counter.id,
                QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
            )
            .order_by(QueueToken.called_at.desc(), QueueToken.id.desc())
            .all()
        )

        if len(active_tokens) > 1:
            raise SnapshotAdapterError(
                f"Counter {counter.id} has multiple active tokens simultaneously: {[t.id for t in active_tokens]}"
            )

        current_token_snapshot: Optional[ServingTokenSnapshot] = None
        if active_tokens:
            token = active_tokens[0]
            if token.id in active_token_ids_across_counters:
                raise SnapshotAdapterError(
                    f"Token {token.id} is associated with multiple counters simultaneously"
                )
            active_token_ids_across_counters.add(token.id)

            if token.service_id != counter.assigned_service_id:
                raise SnapshotAdapterError(
                    f"Token {token.id} service_id {token.service_id} does not match "
                    f"Counter {counter.id} assigned_service_id {counter.assigned_service_id}"
                )

            tok_status = token.status.value if hasattr(token.status, "value") else str(token.status)
            called_at_utc = _ensure_utc(token.called_at, "called_at")
            if called_at_utc is None:
                raise SnapshotAdapterError(f"Token {token.id} has status {tok_status} but null called_at")

            serving_at_utc = None
            if tok_status == "SERVING":
                serving_at_utc = _ensure_utc(token.serving_at, "serving_at")
                if serving_at_utc is None:
                    raise SnapshotAdapterError(f"Token {token.id} has status SERVING but null serving_at")

            current_token_snapshot = ServingTokenSnapshot(
                token_id=token.id,
                token_number=token.token_number,
                service_id=token.service_id,
                counter_id=counter.id,
                status=tok_status,
                called_at_utc=called_at_utc,
                serving_at_utc=serving_at_utc,
            )

        counters_dict[counter.id] = CounterSnapshot(
            counter_id=counter.id,
            office_id=counter.office_id,
            name=counter.name,
            status=status_str,
            status_changed_at_utc=status_changed_at,
            assigned_service_id=counter.assigned_service_id,
            eligible_service_ids=eligible_service_ids,
            current_token=current_token_snapshot,
        )

    # 5. Fetch Queues (Waiting Tokens & Recent Arrivals) for each service
    queues_dict: Dict[int, ServiceQueueSnapshot] = {}
    lookback_60m = snapshot_time_utc - timedelta(minutes=60)

    for service_id in services_dict:
        # A. Current WAITING tokens up to snapshot_time_utc in strict FIFO order
        waiting_rows = (
            db.query(QueueToken)
            .filter(
                QueueToken.office_id == office_id,
                QueueToken.service_id == service_id,
                QueueToken.status == TokenStatus.WAITING,
                QueueToken.created_at <= snapshot_time_utc,
            )
            .order_by(QueueToken.created_at.asc(), QueueToken.id.asc())
            .all()
        )

        waiting_citizens: List[QueueCitizenSnapshot] = []
        for wt in waiting_rows:
            if wt.id in active_token_ids_across_counters:
                raise SnapshotAdapterError(
                    f"Token {wt.id} cannot simultaneously be WAITING and active at a counter"
                )
            arr_time = _ensure_utc(wt.created_at, "created_at")
            if arr_time is None:
                raise SnapshotAdapterError(f"Waiting token {wt.id} has null created_at")
            waiting_citizens.append(
                QueueCitizenSnapshot(
                    token_id=wt.id,
                    token_number=wt.token_number,
                    arrival_time_utc=arr_time,
                )
            )

        # B. Actual historical token creation timestamps in (snapshot_time_utc - 60m, snapshot_time_utc]
        # Any token generated (waiting, served, cancelled, etc.) represents an arrival event
        arrival_rows = (
            db.query(QueueToken.created_at)
            .filter(
                QueueToken.office_id == office_id,
                QueueToken.service_id == service_id,
                QueueToken.created_at > lookback_60m,
                QueueToken.created_at <= snapshot_time_utc,
            )
            .order_by(QueueToken.created_at.asc())
            .all()
        )

        recent_arrivals: List[datetime] = []
        for (arr_dt,) in arrival_rows:
            norm_dt = _ensure_utc(arr_dt, "created_at")
            if norm_dt is not None:
                recent_arrivals.append(norm_dt)

        queues_dict[service_id] = ServiceQueueSnapshot(
            service_id=service_id,
            waiting_tokens=tuple(waiting_citizens),
            recent_arrivals_utc=tuple(recent_arrivals),
        )

    # 6. Construct and validate domain SimulationSnapshot
    return SimulationSnapshot(
        office_id=office.id,
        office_code=office.code,
        snapshot_time_utc=snapshot_time_utc,
        services=services_dict,
        queues=queues_dict,
        counters=counters_dict,
    )
