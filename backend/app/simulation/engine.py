"""
PRAVAAH — Discrete-Event Simulation Engine.

Deterministic, event-driven queue simulator operating entirely on in-memory domain state.
Guarantees:
- Zero database, SQLAlchemy, FastAPI, or global mutable state dependencies.
- Independent FIFO queues maintained per service.
- Individual physical counters with strict eligibility verification.
- Deterministic event loop using priority queue (heapq) with documented tie-breaking.
- Bounded configurable horizon preventing infinite loops.
- Seeded, reproducible stochastic service and arrival processes.
- Clean intervention hook for scheduled counter reassignment and availability changes.
"""

from dataclasses import dataclass
import heapq
import random
from typing import Dict, List, Optional, Set, Tuple

from enum import Enum, unique

from app.simulation.events import (
    EVENT_PRIORITY,
    EventType,
    SimEvent,
)
from app.simulation.processes import (
    BaseArrivalProcess,
    ServiceProcess,
)
from app.simulation.results import (
    InterventionRecord,
    ServiceMetrics,
    SimulationResult,
)
from app.simulation.state import (
    SimCitizen,
    SimCounter,
    SimulationState,
)


@unique
class InterventionPhase(str, Enum):
    REQUESTED = "REQUESTED"
    PENDING_SWITCH_TO_TARGET = "PENDING_SWITCH_TO_TARGET"
    SWITCHING_TO_TARGET = "SWITCHING_TO_TARGET"
    ACTIVE = "ACTIVE"
    PENDING_REVERSION = "PENDING_REVERSION"
    SWITCHING_TO_ORIGINAL = "SWITCHING_TO_ORIGINAL"
    RESTORED = "RESTORED"
    TERMINATED = "TERMINATED"


@dataclass
class SimInterventionRuntime:
    """
    Mutable runtime tracker for an individual counter intervention in the simulator.
    """
    counter_id: int
    original_service_id: int
    target_service_id: int
    requested_time_min: float
    duration_min: Optional[float]
    switching_delay_min: float
    phase: InterventionPhase = InterventionPhase.REQUESTED
    actual_active_start_min: Optional[float] = None
    actual_active_end_min: Optional[float] = None
    actual_restored_min: Optional[float] = None
    actual_active_duration_min: Optional[float] = None
    is_terminated: bool = False


@dataclass(frozen=True)
class CounterReassignIntervention:
    """
    Extension hook for counter reassignment interventions.
    At time_min, counter_id is requested to reassign to new_service_id.
    Supports temporary intervention lifecycle with duration_min and switching_delay_min.
    """
    time_min: float
    counter_id: int
    new_service_id: int
    duration_min: Optional[float] = None
    switching_delay_min: float = 0.0
    original_service_id: Optional[int] = None


@dataclass(frozen=True)
class CounterAvailabilityEvent:
    """
    Extension hook for scheduled counter failure / recovery.
    At time_min, counter_id becomes available (is_available=True) or unavailable (is_available=False).
    announced: bool indicating whether this event is known in advance to the decision world (default: False).
    """
    time_min: float
    counter_id: int
    is_available: bool
    announced: bool = False


class DiscreteEventEngine:
    """
    Pure discrete-event simulation engine for PRAVAAH queue intelligence.
    """
    def __init__(
        self,
        horizon_minutes: float,
        seed: Optional[int] = 42,
        arrival_process: Optional[BaseArrivalProcess] = None,
        service_process: Optional[ServiceProcess] = None,
        interventions: Optional[List[CounterReassignIntervention]] = None,
        availability_events: Optional[List[CounterAvailabilityEvent]] = None,
    ):
        if horizon_minutes <= 0.0:
            raise ValueError(f"horizon_minutes must be > 0.0 (got {horizon_minutes})")

        self.horizon_minutes = float(horizon_minutes)
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.arrival_process = arrival_process
        self.service_process = service_process or ServiceProcess(self._rng)
        self.interventions = list(interventions) if interventions else []
        self.availability_events = list(availability_events) if availability_events else []

    def run(self, initial_state: SimulationState) -> SimulationResult:
        """
        Executes discrete-event simulation up to horizon_minutes on a clone of initial_state.
        The input initial_state is never modified.
        """
        # 1. Deep-clone state to ensure strict run isolation
        state = initial_state.clone()

        # 2. Priority queue & sequence tracker
        event_heap: List[SimEvent] = []
        seq_num = 0

        def schedule_event(time_min: float, event_type: EventType, data: any = None) -> None:
            nonlocal seq_num
            seq_num += 1
            priority = EVENT_PRIORITY[event_type]
            event = SimEvent(
                time=float(time_min),
                priority=priority,
                sequence_num=seq_num,
                event_type=event_type,
                data=data,
            )
            heapq.heappush(event_heap, event)

        # Tracking collections
        completed_citizens: List[SimCitizen] = []
        active_citizens: Dict[int, SimCitizen] = {}  # token_id -> citizen actively being served
        queue_history: List[Tuple[float, int, int]] = []  # (time, service_id, length)

        def record_queue_lengths(t: float) -> None:
            for s_id, q in state.queues.items():
                queue_history.append((t, s_id, q.length()))

        # Helper to dispatch an active idle counter
        def try_dispatch_counter(counter: SimCounter, current_time: float) -> None:
            if not counter.is_active or counter.is_busy or counter.is_switching:
                return

            queue = state.queues.get(counter.assigned_service_id)
            if not queue or not queue.has_waiting():
                return

            citizen = queue.pop_left()
            counter.is_busy = True
            counter.current_token_id = citizen.token_id
            counter.current_service_id = citizen.service_id
            active_citizens[citizen.token_id] = citizen

            citizen.service_start_min = current_time
            citizen.waiting_time_min = current_time - citizen.arrival_time_min
            citizen.counter_id = counter.counter_id

            svc_cfg = state.services[counter.assigned_service_id]
            duration = self.service_process.sample_duration(
                svc_cfg.mean_service_minutes,
                svc_cfg.std_service_minutes,
            )
            completion_time = current_time + duration
            counter.service_end_time_min = completion_time

            schedule_event(
                time_min=completion_time,
                event_type=EventType.SERVICE_COMPLETION,
                data=(counter.counter_id, citizen.token_id),
            )
            record_queue_lengths(current_time)

        # 3. Initialize ongoing tokens from snapshot
        for c_id, counter in sorted(state.counters.items()):
            if c_id in state.initial_serving_tokens:
                serving_snap, elapsed = state.initial_serving_tokens[c_id]
                svc_cfg = state.services[counter.assigned_service_id]

                # Create citizen object for the already active token
                citizen = SimCitizen(
                    token_id=serving_snap.token_id,
                    token_number=serving_snap.token_number,
                    service_id=serving_snap.service_id,
                    arrival_time_min=-elapsed,
                    service_start_min=-elapsed if serving_snap.status == "SERVING" else None,
                    counter_id=c_id,
                )
                active_citizens[citizen.token_id] = citizen
                counter.is_busy = True
                counter.current_token_id = citizen.token_id
                counter.current_service_id = citizen.service_id

                if serving_snap.status == "SERVING":
                    # Elapsed service respected: sample remaining duration
                    rem_duration = self.service_process.sample_remaining_duration(
                        svc_cfg.mean_service_minutes,
                        svc_cfg.std_service_minutes,
                        elapsed=elapsed,
                    )
                    completion_time = rem_duration
                    counter.service_end_time_min = completion_time
                    schedule_event(
                        time_min=completion_time,
                        event_type=EventType.SERVICE_COMPLETION,
                        data=(c_id, citizen.token_id),
                    )
                elif serving_snap.status == "CALLED":
                    # CALLED token starts service immediately at t = 0.0
                    citizen.service_start_min = 0.0
                    citizen.waiting_time_min = 0.0 - citizen.arrival_time_min
                    duration = self.service_process.sample_duration(
                        svc_cfg.mean_service_minutes,
                        svc_cfg.std_service_minutes,
                    )
                    counter.service_end_time_min = duration
                    schedule_event(
                        time_min=duration,
                        event_type=EventType.SERVICE_COMPLETION,
                        data=(c_id, citizen.token_id),
                    )

        # 4. Schedule future arrivals from arrival_process
        if self.arrival_process is not None:
            generated_citizens = self.arrival_process.generate_arrivals(
                horizon_minutes=self.horizon_minutes,
                start_token_id=10000,
            )
            for new_citizen in generated_citizens:
                schedule_event(
                    time_min=new_citizen.arrival_time_min,
                    event_type=EventType.CUSTOMER_ARRIVAL,
                    data=new_citizen,
                )

        # 5. Initialize and validate interventions
        seen_intervention_counters: Set[int] = set()
        active_interventions: Dict[int, SimInterventionRuntime] = {}

        for interv in self.interventions:
            if interv.counter_id in seen_intervention_counters:
                raise ValueError(
                    f"Counter {interv.counter_id} has multiple scheduled interventions. "
                    f"Only one active intervention per counter is permitted."
                )
            seen_intervention_counters.add(interv.counter_id)

            if interv.counter_id not in state.counters:
                raise ValueError(f"Counter {interv.counter_id} does not exist in simulation state")
            counter = state.counters[interv.counter_id]

            if interv.new_service_id not in state.services:
                raise ValueError(f"Target service {interv.new_service_id} does not exist in state")

            if not counter.can_serve(interv.new_service_id):
                raise ValueError(
                    f"Counter {counter.counter_id} ({counter.name}) is not eligible for service {interv.new_service_id} "
                    f"(eligible: {set(counter.eligible_service_ids)})"
                )

            orig_service_id = (
                interv.original_service_id
                if interv.original_service_id is not None
                else counter.assigned_service_id
            )

            if orig_service_id == interv.new_service_id:
                raise ValueError(
                    f"Counter {counter.counter_id} is already assigned to service {interv.new_service_id}"
                )

            if interv.duration_min is not None and interv.duration_min <= 0.0:
                raise ValueError(f"Intervention duration must be > 0.0 minutes (got {interv.duration_min})")

            if interv.switching_delay_min < 0.0:
                raise ValueError(f"Switching delay must be >= 0.0 minutes (got {interv.switching_delay_min})")

            runtime = SimInterventionRuntime(
                counter_id=interv.counter_id,
                original_service_id=orig_service_id,
                target_service_id=interv.new_service_id,
                requested_time_min=interv.time_min,
                duration_min=interv.duration_min,
                switching_delay_min=interv.switching_delay_min,
            )
            active_interventions[interv.counter_id] = runtime

            if 0.0 <= interv.time_min <= self.horizon_minutes:
                schedule_event(
                    time_min=interv.time_min,
                    event_type=EventType.INTERVENTION_REQUEST,
                    data=runtime,
                )

        # 6. Schedule availability events
        for avail in self.availability_events:
            if 0.0 <= avail.time_min <= self.horizon_minutes:
                ev_type = (
                    EventType.COUNTER_AVAILABLE
                    if avail.is_available
                    else EventType.COUNTER_UNAVAILABLE
                )
                schedule_event(
                    time_min=avail.time_min,
                    event_type=ev_type,
                    data=avail.counter_id,
                )

        # Record initial queue state at t = 0.0
        record_queue_lengths(0.0)

        # 7. Schedule initial dispatch of idle counters at t = 0.0 (runs at priority 60, after any t=0 interventions)
        schedule_event(time_min=0.0, event_type=EventType.SERVICE_START, data=None)

        # 8. Main Event Loop
        while event_heap:
            event = heapq.heappop(event_heap)

            # Never process events past the horizon
            if event.time > self.horizon_minutes:
                break

            current_time = event.time

            # A. INTERVENTION_REQUEST
            if event.event_type == EventType.INTERVENTION_REQUEST:
                runtime = event.data
                if runtime.is_terminated:
                    continue

                counter = state.counters[runtime.counter_id]
                if not counter.is_active:
                    runtime.is_terminated = True
                    runtime.phase = InterventionPhase.TERMINATED
                    continue

                if counter.is_busy:
                    # Busy donor counter: current citizen continues normally without interruption!
                    runtime.phase = InterventionPhase.PENDING_SWITCH_TO_TARGET
                else:
                    # Idle donor: enter switching delay immediately
                    if runtime.switching_delay_min > 0.0:
                        runtime.phase = InterventionPhase.SWITCHING_TO_TARGET
                        counter.is_switching = True
                        schedule_event(
                            time_min=current_time + runtime.switching_delay_min,
                            event_type=EventType.INTERVENTION_ACTIVATED,
                            data=runtime,
                        )
                    else:
                        # 0 switching delay -> activate immediately
                        schedule_event(
                            time_min=current_time,
                            event_type=EventType.INTERVENTION_ACTIVATED,
                            data=runtime,
                        )

            # B. INTERVENTION_ACTIVATED (also handles legacy INTERVENTION_REASSIGN)
            elif event.event_type in (EventType.INTERVENTION_ACTIVATED, EventType.INTERVENTION_REASSIGN):
                if isinstance(event.data, tuple):
                    # Legacy hook: (counter_id, new_service_id)
                    counter_id, new_service_id = event.data
                    counter = state.counters[counter_id]
                    if not counter.can_serve(new_service_id):
                        raise ValueError(
                            f"Cannot reassign Counter {counter_id} ({counter.name}) to service {new_service_id}: "
                            f"not in eligible services {set(counter.eligible_service_ids)}"
                        )
                    counter.assigned_service_id = new_service_id
                    if counter.is_active and not counter.is_busy and not counter.is_switching:
                        try_dispatch_counter(counter, current_time)
                else:
                    runtime = event.data
                    if runtime.is_terminated:
                        continue

                    counter = state.counters[runtime.counter_id]
                    if not counter.is_active:
                        runtime.is_terminated = True
                        runtime.phase = InterventionPhase.TERMINATED
                        continue

                    counter.is_switching = False
                    counter.assigned_service_id = runtime.target_service_id
                    runtime.phase = InterventionPhase.ACTIVE
                    runtime.actual_active_start_min = current_time

                    # Schedule duration expiry if temporary duration is configured
                    if runtime.duration_min is not None and runtime.duration_min > 0.0:
                        schedule_event(
                            time_min=current_time + runtime.duration_min,
                            event_type=EventType.INTERVENTION_EXPIRED,
                            data=runtime,
                        )

                    # Counter can immediately serve target service if idle
                    if counter.is_active and not counter.is_busy:
                        try_dispatch_counter(counter, current_time)

            # C. INTERVENTION_EXPIRED
            elif event.event_type == EventType.INTERVENTION_EXPIRED:
                runtime = event.data
                if runtime.is_terminated or runtime.phase != InterventionPhase.ACTIVE:
                    continue

                counter = state.counters[runtime.counter_id]
                if counter.is_busy:
                    # Counter is currently serving a target citizen: finish service first!
                    runtime.phase = InterventionPhase.PENDING_REVERSION
                else:
                    # Idle: begin reversion switching delay immediately
                    runtime.actual_active_end_min = current_time
                    if runtime.switching_delay_min > 0.0:
                        runtime.phase = InterventionPhase.SWITCHING_TO_ORIGINAL
                        counter.is_switching = True
                        schedule_event(
                            time_min=current_time + runtime.switching_delay_min,
                            event_type=EventType.INTERVENTION_RESTORED,
                            data=runtime,
                        )
                    else:
                        schedule_event(
                            time_min=current_time,
                            event_type=EventType.INTERVENTION_RESTORED,
                            data=runtime,
                        )

            # D. INTERVENTION_RESTORED
            elif event.event_type == EventType.INTERVENTION_RESTORED:
                runtime = event.data
                if runtime.is_terminated:
                    continue

                counter = state.counters[runtime.counter_id]
                counter.is_switching = False
                counter.assigned_service_id = runtime.original_service_id
                runtime.phase = InterventionPhase.RESTORED
                runtime.actual_restored_min = current_time
                if runtime.actual_active_start_min is not None and runtime.actual_active_end_min is not None:
                    runtime.actual_active_duration_min = (
                        runtime.actual_active_end_min - runtime.actual_active_start_min
                    )

                # Counter can immediately serve its original queue if active and idle
                if counter.is_active and not counter.is_busy:
                    try_dispatch_counter(counter, current_time)

            # E. COUNTER_UNAVAILABLE
            elif event.event_type == EventType.COUNTER_UNAVAILABLE:
                counter_id = event.data
                counter = state.counters[counter_id]
                counter.is_active = False
                counter.is_switching = False

                # Terminate any active intervention on this counter and restore original assignment
                if counter_id in active_interventions:
                    runtime = active_interventions[counter_id]
                    if not runtime.is_terminated and runtime.phase not in (
                        InterventionPhase.RESTORED,
                        InterventionPhase.TERMINATED,
                    ):
                        runtime.is_terminated = True
                        runtime.phase = InterventionPhase.TERMINATED
                        if runtime.actual_active_start_min is not None and runtime.actual_active_end_min is None:
                            runtime.actual_active_end_min = current_time
                        counter.assigned_service_id = runtime.original_service_id

            # F. COUNTER_AVAILABLE
            elif event.event_type == EventType.COUNTER_AVAILABLE:
                counter_id = event.data
                counter = state.counters[counter_id]
                counter.is_active = True
                counter.is_switching = False
                if not counter.is_busy:
                    try_dispatch_counter(counter, current_time)

            # G. SERVICE_COMPLETION
            elif event.event_type == EventType.SERVICE_COMPLETION:
                counter_id, token_id = event.data
                counter = state.counters[counter_id]
                counter.is_busy = False
                counter.current_token_id = None
                counter.current_service_id = None
                counter.service_end_time_min = None

                citizen = active_citizens.pop(token_id, None)
                if citizen:
                    citizen.service_completion_min = current_time
                    completed_citizens.append(citizen)

                # Check if this counter has an intervention with a pending phase change
                handled_by_intervention = False
                if counter_id in active_interventions:
                    runtime = active_interventions[counter_id]
                    if runtime.phase == InterventionPhase.PENDING_SWITCH_TO_TARGET:
                        handled_by_intervention = True
                        if counter.is_active:
                            if runtime.switching_delay_min > 0.0:
                                runtime.phase = InterventionPhase.SWITCHING_TO_TARGET
                                counter.is_switching = True
                                schedule_event(
                                    time_min=current_time + runtime.switching_delay_min,
                                    event_type=EventType.INTERVENTION_ACTIVATED,
                                    data=runtime,
                                )
                            else:
                                schedule_event(
                                    time_min=current_time,
                                    event_type=EventType.INTERVENTION_ACTIVATED,
                                    data=runtime,
                                )

                    elif runtime.phase == InterventionPhase.PENDING_REVERSION:
                        handled_by_intervention = True
                        runtime.actual_active_end_min = current_time
                        if counter.is_active:
                            if runtime.switching_delay_min > 0.0:
                                runtime.phase = InterventionPhase.SWITCHING_TO_ORIGINAL
                                counter.is_switching = True
                                schedule_event(
                                    time_min=current_time + runtime.switching_delay_min,
                                    event_type=EventType.INTERVENTION_RESTORED,
                                    data=runtime,
                                )
                            else:
                                schedule_event(
                                    time_min=current_time,
                                    event_type=EventType.INTERVENTION_RESTORED,
                                    data=runtime,
                                )

                if not handled_by_intervention:
                    if counter.is_active and not counter.is_switching:
                        try_dispatch_counter(counter, current_time)

            # H. CUSTOMER_ARRIVAL
            elif event.event_type == EventType.CUSTOMER_ARRIVAL:
                citizen = event.data
                queue = state.queues[citizen.service_id]

                # Check if there is an active idle counter assigned to this service (not switching)
                eligible_idle = [
                    c for c in sorted(state.counters.values(), key=lambda c: c.counter_id)
                    if c.is_active and not c.is_busy and not c.is_switching and c.assigned_service_id == citizen.service_id
                ]

                if eligible_idle and not queue.has_waiting():
                    # Direct service start
                    counter = eligible_idle[0]
                    counter.is_busy = True
                    counter.current_token_id = citizen.token_id
                    counter.current_service_id = citizen.service_id
                    active_citizens[citizen.token_id] = citizen

                    citizen.service_start_min = current_time
                    citizen.waiting_time_min = current_time - citizen.arrival_time_min
                    citizen.counter_id = counter.counter_id

                    svc_cfg = state.services[citizen.service_id]
                    duration = self.service_process.sample_duration(
                        svc_cfg.mean_service_minutes,
                        svc_cfg.std_service_minutes,
                    )
                    completion_time = current_time + duration
                    counter.service_end_time_min = completion_time

                    schedule_event(
                        time_min=completion_time,
                        event_type=EventType.SERVICE_COMPLETION,
                        data=(counter.counter_id, citizen.token_id),
                    )
                else:
                    queue.append(citizen)

                record_queue_lengths(current_time)

            # I. SERVICE_START (dispatches any eligible idle counters)
            elif event.event_type == EventType.SERVICE_START:
                for counter in sorted(state.counters.values(), key=lambda c: c.counter_id):
                    if counter.is_active and not counter.is_busy and not counter.is_switching:
                        try_dispatch_counter(counter, current_time)

        # 9. Compute Performance Metrics
        service_metrics: Dict[int, ServiceMetrics] = {}
        total_served_wait = 0.0
        total_unserved_wait = 0.0
        all_served_waits: List[float] = []

        # Served citizens grouped by service
        served_by_service: Dict[int, List[SimCitizen]] = {s_id: [] for s_id in state.services}
        for sc in completed_citizens:
            if sc.waiting_time_min is not None:
                served_by_service[sc.service_id].append(sc)
                total_served_wait += sc.waiting_time_min
                all_served_waits.append(sc.waiting_time_min)

        # Unserved citizens remaining in queues at horizon
        unserved_ids: List[int] = []
        for s_id, s_queue in state.queues.items():
            served_list = served_by_service.get(s_id, [])
            served_count = len(served_list)
            served_wait = sum(c.waiting_time_min for c in served_list if c.waiting_time_min is not None)

            # Calculate backlog accrued wait for citizens still waiting at horizon
            unserved_wait = 0.0
            unserved_count = s_queue.length()
            for uc in s_queue.waiting_citizens:
                unserved_ids.append(uc.token_id)
                # Accrued wait up to horizon
                accrued = max(0.0, self.horizon_minutes - uc.arrival_time_min)
                unserved_wait += accrued

            total_unserved_wait += unserved_wait

            avg_wait = (served_wait / served_count) if served_count > 0 else 0.0
            max_wait = max((c.waiting_time_min for c in served_list if c.waiting_time_min is not None), default=0.0)
            svc_cfg = state.services[s_id]

            service_metrics[s_id] = ServiceMetrics(
                service_id=s_id,
                service_code=svc_cfg.service_code,
                customers_served=served_count,
                waiting_person_minutes=served_wait,
                unserved_customers_count=unserved_count,
                unserved_waiting_person_minutes=unserved_wait,
                total_waiting_person_minutes=served_wait + unserved_wait,
                avg_waiting_time_minutes=avg_wait,
                max_waiting_time_minutes=max_wait,
            )

        total_customers_served = len(completed_citizens)
        overall_avg_wait = (
            sum(all_served_waits) / len(all_served_waits) if all_served_waits else 0.0
        )
        overall_max_wait = max(all_served_waits, default=0.0)

        service_waiting_times = {
            s_id: tuple(
                [c.waiting_time_min for c in served_by_service[s_id] if c.waiting_time_min is not None]
                + [max(0.0, self.horizon_minutes - uc.arrival_time_min) for uc in state.queues[s_id].waiting_citizens]
            )
            for s_id in state.services
        }

        # Build telemetry records for interventions
        intervention_records = tuple(
            InterventionRecord(
                counter_id=r.counter_id,
                original_service_id=r.original_service_id,
                target_service_id=r.target_service_id,
                requested_time_min=r.requested_time_min,
                duration_min=r.duration_min,
                switching_delay_min=r.switching_delay_min,
                actual_active_start_min=r.actual_active_start_min,
                actual_active_end_min=r.actual_active_end_min,
                actual_restored_min=r.actual_restored_min,
                actual_active_duration_min=r.actual_active_duration_min,
                status=r.phase.value,
            )
            for r in active_interventions.values()
        )

        return SimulationResult(
            horizon_minutes=self.horizon_minutes,
            total_customers_served=total_customers_served,
            total_waiting_person_minutes=total_served_wait + total_unserved_wait,
            served_waiting_person_minutes=total_served_wait,
            unserved_waiting_person_minutes=total_unserved_wait,
            avg_waiting_time_minutes=overall_avg_wait,
            max_waiting_time_minutes=overall_max_wait,
            service_metrics=service_metrics,
            queue_length_history=tuple(queue_history),
            completed_citizen_ids=tuple(c.token_id for c in completed_citizens),
            unserved_citizen_ids=tuple(unserved_ids),
            service_waiting_times=service_waiting_times,
            intervention_records=intervention_records,
        )
