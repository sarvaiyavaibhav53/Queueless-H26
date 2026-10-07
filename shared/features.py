"""
PRAVAAH — Core Simulation Snapshot & Shared Raw Feature Contract.

Architectural Guarantees:
- Pure, deterministic, framework-agnostic implementation.
- Standard-library only (no SQLAlchemy, no pandas, no scikit-learn, no FastAPI).
- Timezone-aware UTC canonical timestamps; local time derived via Asia/Kolkata.
- Zero system clock calls (no datetime.now() or utcnow()).
- Finite traffic intensity sentinel (99.0) under zero capacity with positive demand.
- ML Encoding Boundary:
    RawFeatureVector (Domain Level)
            ↓
    future ML Preprocessing (Pipelines / ColumnTransformers)
            ↓
    OneHotEncoder / Numerical Scalers
            ↓
    Estimators / Regression Models
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, FrozenSet, Optional, Tuple, Union
import math

# Canonical Module-Level Timezone Constants
UTC_TIMEZONE = timezone.utc
LOCAL_TIMEZONE = ZoneInfo("Asia/Kolkata")

# Canonical Immutable Feature Sequence for downstream ML pipelines
RAW_FEATURE_NAMES: Tuple[str, ...] = (
    "office_id",
    "service_id",
    "hour",
    "day_of_week",
    "people_ahead",
    "queue_length",
    "assigned_active_counters",
    "busy_assigned_counters",
    "max_elapsed_service_minutes",
    "service_mean_minutes",
    "service_std_minutes",
    "arrival_rate_30m",
    "arrival_rate_60m",
    "service_capacity_mu",
    "traffic_intensity",
)


def _validate_utc_datetime(dt: Optional[datetime], field_name: str) -> None:
    """Ensures datetime values are timezone-aware and set to UTC."""
    if dt is None:
        return
    if not isinstance(dt, datetime):
        raise TypeError(f"{field_name} must be a datetime instance (got {type(dt)})")
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime (got {dt})")


# =====================================================================
# 1. Snapshot Dataclasses
# =====================================================================

@dataclass(frozen=True)
class QueueCitizenSnapshot:
    """
    Represents an individual citizen waiting in a specific service FIFO queue.
    service_id is not stored here as it is uniquely provided by the parent ServiceQueueSnapshot.
    """
    token_id: int
    token_number: str
    arrival_time_utc: datetime

    def __post_init__(self) -> None:
        if self.token_id <= 0:
            raise ValueError(f"token_id must be a positive integer (got {self.token_id})")
        if not self.token_number or not self.token_number.strip():
            raise ValueError("token_number must be a non-empty string")
        _validate_utc_datetime(self.arrival_time_utc, "arrival_time_utc")


@dataclass(frozen=True)
class ServingTokenSnapshot:
    """
    Represents a token currently actively engaged with a service counter.
    Allowed statuses: 'CALLED', 'SERVING'.
    """
    token_id: int
    token_number: str
    service_id: int
    counter_id: int
    status: str
    called_at_utc: datetime
    serving_at_utc: Optional[datetime]

    def __post_init__(self) -> None:
        if self.token_id <= 0:
            raise ValueError(f"token_id must be a positive integer (got {self.token_id})")
        if not self.token_number or not self.token_number.strip():
            raise ValueError("token_number must be a non-empty string")
        if self.service_id <= 0:
            raise ValueError(f"service_id must be a positive integer (got {self.service_id})")
        if self.counter_id <= 0:
            raise ValueError(f"counter_id must be a positive integer (got {self.counter_id})")
        if self.status not in ("CALLED", "SERVING"):
            raise ValueError(f"status must be 'CALLED' or 'SERVING' (got '{self.status}')")
        
        _validate_utc_datetime(self.called_at_utc, "called_at_utc")
        _validate_utc_datetime(self.serving_at_utc, "serving_at_utc")

        if self.status == "SERVING":
            if self.serving_at_utc is None:
                raise ValueError("serving_at_utc is required when status is 'SERVING'")
            if self.serving_at_utc < self.called_at_utc:
                raise ValueError("serving_at_utc cannot be earlier than called_at_utc")
        elif self.status == "CALLED":
            if self.serving_at_utc is not None:
                raise ValueError("serving_at_utc must be None when status is 'CALLED'")


@dataclass(frozen=True)
class CounterSnapshot:
    """
    Represents a physical counter desk entity.
    Allowed statuses: 'ACTIVE', 'UNAVAILABLE'.
    """
    counter_id: int
    office_id: int
    name: str
    status: str
    status_changed_at_utc: datetime
    assigned_service_id: int
    eligible_service_ids: FrozenSet[int]
    current_token: Optional[ServingTokenSnapshot]

    def __post_init__(self) -> None:
        if self.counter_id <= 0:
            raise ValueError(f"counter_id must be a positive integer (got {self.counter_id})")
        if self.office_id <= 0:
            raise ValueError(f"office_id must be a positive integer (got {self.office_id})")
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.status not in ("ACTIVE", "UNAVAILABLE"):
            raise ValueError(f"status must be 'ACTIVE' or 'UNAVAILABLE' (got '{self.status}')")
        
        _validate_utc_datetime(self.status_changed_at_utc, "status_changed_at_utc")

        if not self.eligible_service_ids:
            raise ValueError("eligible_service_ids cannot be empty")
        if self.assigned_service_id not in self.eligible_service_ids:
            raise ValueError(
                f"assigned_service_id ({self.assigned_service_id}) must be in eligible_service_ids "
                f"({set(self.eligible_service_ids)})"
            )

        if self.current_token is not None:
            if self.current_token.counter_id != self.counter_id:
                raise ValueError(
                    f"current_token.counter_id ({self.current_token.counter_id}) must match "
                    f"counter_id ({self.counter_id})"
                )
            if self.current_token.service_id != self.assigned_service_id:
                raise ValueError(
                    f"current_token.service_id ({self.current_token.service_id}) must match "
                    f"assigned_service_id ({self.assigned_service_id})"
                )

    @property
    def is_busy(self) -> bool:
        """Counter is busy if actively associated with a CALLED or SERVING token."""
        return self.current_token is not None

    @property
    def is_available(self) -> bool:
        """Counter is available if ACTIVE and has no active token."""
        return self.status == "ACTIVE" and self.current_token is None

    def elapsed_service_minutes(self, snapshot_time_utc: datetime) -> float:
        """
        Calculates elapsed serving minutes up to snapshot_time_utc.
        Rules:
        - Idle -> 0.0
        - CALLED -> 0.0
        - SERVING -> difference between snapshot_time_utc and serving_at_utc
        - Clamps negative values to 0.0
        """
        _validate_utc_datetime(snapshot_time_utc, "snapshot_time_utc")
        if self.current_token is None or self.current_token.status != "SERVING":
            return 0.0
        if self.current_token.serving_at_utc is None:
            return 0.0

        elapsed_sec = (snapshot_time_utc - self.current_token.serving_at_utc).total_seconds()
        return max(0.0, elapsed_sec / 60.0)


@dataclass(frozen=True)
class ServiceConfigSnapshot:
    """
    Configuration parameters defining service processing times.
    """
    service_id: int
    service_code: str
    service_name: str
    mean_service_minutes: float
    std_service_minutes: float

    def __post_init__(self) -> None:
        if self.service_id <= 0:
            raise ValueError(f"service_id must be a positive integer (got {self.service_id})")
        if not self.service_code or not self.service_code.strip():
            raise ValueError("service_code must be a non-empty string")
        if not self.service_name or not self.service_name.strip():
            raise ValueError("service_name must be a non-empty string")
        if not math.isfinite(self.mean_service_minutes) or self.mean_service_minutes <= 0.0:
            raise ValueError(f"mean_service_minutes must be finite and > 0.0 (got {self.mean_service_minutes})")
        if not math.isfinite(self.std_service_minutes) or self.std_service_minutes < 0.0:
            raise ValueError(f"std_service_minutes must be finite and >= 0.0 (got {self.std_service_minutes})")


@dataclass(frozen=True)
class ServiceQueueSnapshot:
    """
    Per-service queue state and recent arrival history.
    waiting_tokens must strictly maintain FIFO order by arrival_time_utc.
    """
    service_id: int
    waiting_tokens: Tuple[QueueCitizenSnapshot, ...] = ()
    recent_arrivals_utc: Tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        if self.service_id <= 0:
            raise ValueError(f"service_id must be a positive integer (got {self.service_id})")

        # Invariant 2: FIFO order check (arrival_time_utc ascending)
        for i in range(len(self.waiting_tokens) - 1):
            t1 = self.waiting_tokens[i].arrival_time_utc
            t2 = self.waiting_tokens[i + 1].arrival_time_utc
            if t1 > t2:
                raise ValueError(
                    f"waiting_tokens must be in strict FIFO order by arrival_time_utc. "
                    f"Index {i} ({t1}) is after index {i+1} ({t2})"
                )

        for arr in self.recent_arrivals_utc:
            _validate_utc_datetime(arr, "recent_arrivals_utc entry")


@dataclass(frozen=True)
class SimulationSnapshot:
    """
    Self-contained, serializable state snapshot of an office at point-in-time t.
    """
    office_id: int
    office_code: str
    snapshot_time_utc: datetime
    services: Dict[int, ServiceConfigSnapshot]
    queues: Dict[int, ServiceQueueSnapshot]
    counters: Dict[int, CounterSnapshot]

    def __post_init__(self) -> None:
        if self.office_id <= 0:
            raise ValueError(f"office_id must be a positive integer (got {self.office_id})")
        if not self.office_code or not self.office_code.strip():
            raise ValueError("office_code must be a non-empty string")
        _validate_utc_datetime(self.snapshot_time_utc, "snapshot_time_utc")

        # Invariant checks across entities
        seen_token_ids = set()

        # Queues consistency
        for s_id, queue in self.queues.items():
            if queue.service_id != s_id:
                raise ValueError(f"Queue key {s_id} does not match queue.service_id {queue.service_id}")
            if s_id not in self.services:
                raise ValueError(f"Queue service_id {s_id} is not registered in services configuration")

            for citizen in queue.waiting_tokens:
                if citizen.token_id in seen_token_ids:
                    raise ValueError(f"Duplicate token_id {citizen.token_id} found in waiting queues")
                seen_token_ids.add(citizen.token_id)

                if citizen.arrival_time_utc > self.snapshot_time_utc:
                    raise ValueError(
                        f"Waiting token {citizen.token_id} has arrival time {citizen.arrival_time_utc} "
                        f"in the future relative to snapshot_time_utc {self.snapshot_time_utc}"
                    )

            for arr in queue.recent_arrivals_utc:
                if arr > self.snapshot_time_utc:
                    raise ValueError(
                        f"Recent arrival timestamp {arr} is in the future relative to "
                        f"snapshot_time_utc {self.snapshot_time_utc}"
                    )

        # Counters consistency
        for c_id, counter in self.counters.items():
            if counter.counter_id != c_id:
                raise ValueError(f"Counter key {c_id} does not match counter.counter_id {counter.counter_id}")
            if counter.office_id != self.office_id:
                raise ValueError(
                    f"Counter {c_id} belongs to office_id {counter.office_id}, expected {self.office_id}"
                )
            if counter.assigned_service_id not in self.services:
                raise ValueError(
                    f"Counter {c_id} assigned_service_id {counter.assigned_service_id} "
                    f"is not registered in services configuration"
                )

            # Invariant 5: A waiting token must not also be active at a counter
            if counter.current_token is not None:
                token_id = counter.current_token.token_id
                if token_id in seen_token_ids:
                    raise ValueError(
                        f"Token {token_id} cannot simultaneously be waiting in a queue and active at Counter {c_id}"
                    )
                seen_token_ids.add(token_id)

    @property
    def snapshot_time_local(self) -> datetime:
        """Interprets snapshot time in Asia/Kolkata timezone."""
        return self.snapshot_time_utc.astimezone(LOCAL_TIMEZONE)


# =====================================================================
# 2. Raw Feature Vector
# =====================================================================

@dataclass(frozen=True)
class RawFeatureVector:
    """
    Immutable typed domain feature vector produced by compute_raw_features().
    These are unencoded domain features in natural units.
    """
    office_id: str                      # String code: e.g., "AMD"
    service_id: str                     # String code: e.g., "PT"
    hour: int                           # Local hour in Asia/Kolkata: 0..23
    day_of_week: int                    # Local weekday index: 0..6 (0=Monday)
    people_ahead: int                   # Count of citizens waiting ahead in FIFO
    queue_length: int                   # Total citizens waiting in this service queue
    assigned_active_counters: int       # Number of ACTIVE counters assigned to this service
    busy_assigned_counters: int          # Active assigned counters currently busy (CALLED or SERVING)
    max_elapsed_service_minutes: float  # Maximum elapsed minutes among SERVING active counters (0.0 if none)
    service_mean_minutes: float         # Historical mean service duration
    service_std_minutes: float          # Historical standard deviation of duration
    arrival_rate_30m: float             # Trailing 30m arrival rate (citizens/hour)
    arrival_rate_60m: float             # Trailing 60m arrival rate (citizens/hour)
    service_capacity_mu: float          # Capacity throughput mu (citizens/hour)
    traffic_intensity: float            # Finite ratio lambda_60 / mu (capped at 99.0 if mu=0, lambda>0)

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """Exports raw features in canonical order."""
        return {name: getattr(self, name) for name in RAW_FEATURE_NAMES}


def raw_features_to_dict(features: RawFeatureVector) -> Dict[str, Union[str, int, float]]:
    """Helper function to export RawFeatureVector to a plain dictionary."""
    return features.to_dict()


# =====================================================================
# 3. Pure Feature Calculation Function
# =====================================================================

def compute_raw_features(
    snapshot: SimulationSnapshot,
    service_id: int,
    position: int
) -> RawFeatureVector:
    """
    Pure, deterministic feature calculation function.
    
    Args:
        snapshot: SimulationSnapshot instance at prediction instant t.
        service_id: Service identifier being evaluated.
        position: 1-based citizen queue position in this service's queue.
                  (position=1 means at the head; for newly arriving citizens,
                   position = len(waiting_tokens) + 1).

    Returns:
        RawFeatureVector with all raw domain feature values.
    """
    if position < 1:
        raise ValueError(f"Position must be a positive 1-based integer (got {position})")
    if service_id not in snapshot.services:
        raise KeyError(f"service_id {service_id} not found in snapshot services")
    if service_id not in snapshot.queues:
        raise KeyError(f"service_id {service_id} not found in snapshot queues")

    service_cfg = snapshot.services[service_id]
    queue = snapshot.queues[service_id]
    t_snap = snapshot.snapshot_time_utc

    # 1. Identity / Context
    office_id_code = snapshot.office_code
    service_id_code = service_cfg.service_code
    hour = snapshot.snapshot_time_local.hour
    day_of_week = snapshot.snapshot_time_local.weekday()

    # 2. Queue State
    people_ahead = position - 1
    queue_length = len(queue.waiting_tokens)

    # 3. Counter Fleet Metrics
    active_assigned_counters = [
        c for c in snapshot.counters.values()
        if c.status == "ACTIVE" and c.assigned_service_id == service_id
    ]
    assigned_active_count = len(active_assigned_counters)
    busy_assigned_count = sum(1 for c in active_assigned_counters if c.is_busy)

    # 4. Longest Ongoing Service Progress
    serving_elapsed = [
        c.elapsed_service_minutes(t_snap)
        for c in active_assigned_counters
        if c.current_token is not None and c.current_token.status == "SERVING"
    ]
    max_elapsed_service_minutes = max(serving_elapsed) if serving_elapsed else 0.0

    # 5. Service Characteristics
    service_mean = service_cfg.mean_service_minutes
    service_std = service_cfg.std_service_minutes

    # 6. Observed Arrival Rates: (t_snap - W, t_snap]
    t_30 = t_snap - timedelta(minutes=30)
    t_60 = t_snap - timedelta(minutes=60)

    count_30 = sum(1 for t_arr in queue.recent_arrivals_utc if t_30 < t_arr <= t_snap)
    count_60 = sum(1 for t_arr in queue.recent_arrivals_utc if t_60 < t_arr <= t_snap)

    arrival_rate_30m = count_30 / 0.5   # arrivals / 0.5 hr -> citizens/hr
    arrival_rate_60m = count_60 / 1.0   # arrivals / 1.0 hr -> citizens/hr

    # 7. Service Capacity mu
    if assigned_active_count > 0:
        service_capacity_mu = assigned_active_count * (60.0 / service_mean)
    else:
        service_capacity_mu = 0.0

    # 8. Traffic Intensity rho (Locked Finite Behavior)
    if service_capacity_mu > 0.0:
        traffic_intensity = arrival_rate_60m / service_capacity_mu
    elif arrival_rate_60m == 0.0:
        traffic_intensity = 0.0
    else:
        traffic_intensity = 99.0

    # Sanity verification against non-finite output
    if not math.isfinite(traffic_intensity):
        raise RuntimeError("traffic_intensity produced a non-finite value")

    return RawFeatureVector(
        office_id=office_id_code,
        service_id=service_id_code,
        hour=hour,
        day_of_week=day_of_week,
        people_ahead=people_ahead,
        queue_length=queue_length,
        assigned_active_counters=assigned_active_count,
        busy_assigned_counters=busy_assigned_count,
        max_elapsed_service_minutes=max_elapsed_service_minutes,
        service_mean_minutes=service_mean,
        service_std_minutes=service_std,
        arrival_rate_30m=arrival_rate_30m,
        arrival_rate_60m=arrival_rate_60m,
        service_capacity_mu=service_capacity_mu,
        traffic_intensity=traffic_intensity,
    )
