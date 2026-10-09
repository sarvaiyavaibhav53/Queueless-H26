"""
PRAVAAH — Simulation Domain State.

Pure Python domain objects representing simulation state derived from SimulationSnapshot.
Guarantees:
- Zero external database or framework dependencies.
- Completely clonable / decoupled state for isolated scenario evaluations.
- Preserves individual physical counter assignments and service eligibility.
- Preserves independent FIFO queues per service.
- Strict relative simulation time in minutes from snapshot moment (t = 0.0).
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Tuple

from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    SimulationSnapshot,
)


@dataclass
class SimCitizen:
    """
    Simulated citizen holding a queue token.
    arrival_time_min is expressed in simulation minutes relative to snapshot start (t = 0.0).
    Negative arrival_time_min denotes arrival prior to snapshot moment.
    """
    token_id: int
    token_number: str
    service_id: int
    arrival_time_min: float
    service_start_min: Optional[float] = None
    service_completion_min: Optional[float] = None
    counter_id: Optional[int] = None
    waiting_time_min: Optional[float] = None

    def clone(self) -> "SimCitizen":
        return SimCitizen(
            token_id=self.token_id,
            token_number=self.token_number,
            service_id=self.service_id,
            arrival_time_min=self.arrival_time_min,
            service_start_min=self.service_start_min,
            service_completion_min=self.service_completion_min,
            counter_id=self.counter_id,
            waiting_time_min=self.waiting_time_min,
        )


@dataclass
class SimCounter:
    """
    Individual physical counter desk.
    """
    counter_id: int
    office_id: int
    name: str
    assigned_service_id: int
    eligible_service_ids: FrozenSet[int]
    is_active: bool = True
    is_busy: bool = False
    is_switching: bool = False
    current_token_id: Optional[int] = None
    current_service_id: Optional[int] = None
    service_end_time_min: Optional[float] = None

    def can_serve(self, service_id: int) -> bool:
        """A counter can only serve a service for which it is eligible."""
        return service_id in self.eligible_service_ids

    def clone(self) -> "SimCounter":
        return SimCounter(
            counter_id=self.counter_id,
            office_id=self.office_id,
            name=self.name,
            assigned_service_id=self.assigned_service_id,
            eligible_service_ids=self.eligible_service_ids,
            is_active=self.is_active,
            is_busy=self.is_busy,
            is_switching=self.is_switching,
            current_token_id=self.current_token_id,
            current_service_id=self.current_service_id,
            service_end_time_min=self.service_end_time_min,
        )


@dataclass
class SimQueue:
    """
    Single FIFO queue for an individual service.
    """
    service_id: int
    waiting_citizens: List[SimCitizen] = field(default_factory=list)

    def has_waiting(self) -> bool:
        return len(self.waiting_citizens) > 0

    def pop_left(self) -> SimCitizen:
        if not self.waiting_citizens:
            raise IndexError(f"Queue for service {self.service_id} is empty")
        return self.waiting_citizens.pop(0)

    def append(self, citizen: SimCitizen) -> None:
        if citizen.service_id != self.service_id:
            raise ValueError(
                f"Citizen service_id {citizen.service_id} does not match queue service_id {self.service_id}"
            )
        self.waiting_citizens.append(citizen)

    def length(self) -> int:
        return len(self.waiting_citizens)

    def clone(self) -> "SimQueue":
        return SimQueue(
            service_id=self.service_id,
            waiting_citizens=[c.clone() for c in self.waiting_citizens],
        )


@dataclass
class SimulationState:
    """
    Complete in-memory state of the simulated office.
    """
    office_id: int
    office_code: str
    snapshot_time_utc: datetime
    services: Dict[int, ServiceConfigSnapshot]
    queues: Dict[int, SimQueue]
    counters: Dict[int, SimCounter]
    # Existing active tokens from snapshot: counter_id -> (ServingTokenSnapshot, elapsed_minutes)
    initial_serving_tokens: Dict[int, Tuple[ServingTokenSnapshot, float]] = field(default_factory=dict)
    # Current traffic intensity rho (lambda / mu) per service from current state
    traffic_intensity_by_service: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: SimulationSnapshot) -> "SimulationState":
        """
        Constructs an independent, mutable SimulationState from a frozen SimulationSnapshot.
        Simulation time starts at t = 0.0 minutes corresponding to snapshot_time_utc.
        """
        # 1. Services
        services_map = dict(snapshot.services)

        # 2. Queues
        queues_map: Dict[int, SimQueue] = {}
        for s_id, s_queue in snapshot.queues.items():
            citizens_list: List[SimCitizen] = []
            for w_token in s_queue.waiting_tokens:
                # Arrival time in minutes relative to snapshot_time_utc (<= 0.0)
                diff_sec = (w_token.arrival_time_utc - snapshot.snapshot_time_utc).total_seconds()
                arr_min = diff_sec / 60.0
                citizens_list.append(
                    SimCitizen(
                        token_id=w_token.token_id,
                        token_number=w_token.token_number,
                        service_id=s_id,
                        arrival_time_min=arr_min,
                    )
                )
            queues_map[s_id] = SimQueue(service_id=s_id, waiting_citizens=citizens_list)

        # 3. Counters & active tokens
        counters_map: Dict[int, SimCounter] = {}
        initial_active: Dict[int, Tuple[ServingTokenSnapshot, float]] = {}

        for c_id, c_snap in snapshot.counters.items():
            is_active = (c_snap.status == "ACTIVE")
            is_busy = c_snap.is_busy
            curr_token_id = c_snap.current_token.token_id if c_snap.current_token else None
            curr_service_id = c_snap.current_token.service_id if c_snap.current_token else None

            if c_snap.current_token is not None:
                elapsed = c_snap.elapsed_service_minutes(snapshot.snapshot_time_utc)
                initial_active[c_id] = (c_snap.current_token, elapsed)

            counters_map[c_id] = SimCounter(
                counter_id=c_id,
                office_id=c_snap.office_id,
                name=c_snap.name,
                assigned_service_id=c_snap.assigned_service_id,
                eligible_service_ids=c_snap.eligible_service_ids,
                is_active=is_active,
                is_busy=is_busy,
                current_token_id=curr_token_id,
                current_service_id=curr_service_id,
                service_end_time_min=None,
            )

        # 4. Compute current-state traffic intensity rho per service from snapshot
        rho_by_service: Dict[int, float] = {}
        for s_id, s_queue in snapshot.queues.items():
            count_60 = sum(
                1 for arr_time in s_queue.recent_arrivals_utc
                if 0.0 <= (snapshot.snapshot_time_utc - arr_time).total_seconds() / 60.0 <= 60.0
            )
            svc = services_map.get(s_id)
            if svc and svc.mean_service_minutes > 0.0:
                active_count = sum(
                    1 for c in snapshot.counters.values()
                    if c.status == "ACTIVE" and c.assigned_service_id == s_id
                )
                mu = active_count * (60.0 / svc.mean_service_minutes)
                if mu > 0.0:
                    rho_by_service[s_id] = count_60 / mu
                elif count_60 == 0:
                    rho_by_service[s_id] = 0.0
                else:
                    rho_by_service[s_id] = 99.0
            else:
                rho_by_service[s_id] = 0.0

        return cls(
            office_id=snapshot.office_id,
            office_code=snapshot.office_code,
            snapshot_time_utc=snapshot.snapshot_time_utc,
            services=services_map,
            queues=queues_map,
            counters=counters_map,
            initial_serving_tokens=initial_active,
            traffic_intensity_by_service=rho_by_service,
        )

    def clone(self) -> "SimulationState":
        """
        Deep clone to ensure multiple candidate simulation runs are strictly isolated.
        """
        return SimulationState(
            office_id=self.office_id,
            office_code=self.office_code,
            snapshot_time_utc=self.snapshot_time_utc,
            services=dict(self.services),
            queues={s_id: q.clone() for s_id, q in self.queues.items()},
            counters={c_id: c.clone() for c_id, c in self.counters.items()},
            initial_serving_tokens=dict(self.initial_serving_tokens),
            traffic_intensity_by_service=dict(self.traffic_intensity_by_service),
        )
