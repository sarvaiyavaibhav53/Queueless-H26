"""
PRAVAAH — Candidate Intervention Generator & Naive Policy Builder.

Generates bounded, valid counter reassignment candidates and constructs the naive baseline.
Guarantees:
- Never assigns a counter to an ineligible service.
- Never reassigns an inactive/unavailable counter.
- Never reassigns a counter to its currently assigned service.
- Prefers idle counters and preserves active customer service without interruption.
- Bounded search space respecting max_candidates.
"""

from typing import Dict, List, Optional, Sequence

from app.interventions.models import InterventionCandidate
from app.simulation.state import SimCounter, SimulationState
from shared.stability import is_stable


def build_naive_intervention(
    state: SimulationState,
    traffic_intensity_by_service: Optional[Dict[int, float]] = None,
    duration_min: float = 30.0,
    switching_delay_min: float = 3.0,
) -> Optional[InterventionCandidate]:
    """
    Constructs the NAIVE heuristic policy using only current-state information:
    a. Target service = service with the longest current waiting queue.
    b. If target service is STABLE (rho < 0.85) or all queues empty, return no action.
    c. Donor = eligible active counter satisfying:
       - counter is ACTIVE
       - counter is not already assigned to target service
       - counter is eligible to serve target service
       - donor's source service keeps at least 1 active assigned counter after reassignment
    d. Ranked deterministically:
       1. lowest current rho of donor's source service
       2. idle donor preferred on rho tie
       3. lowest counter_id as final tie-break
    e. If no valid donor exists, return no action.
    f. Naive uses a temporary reassignment (default 30.0m duration, 3.0m switching delay).
    """
    rho_map = (
        traffic_intensity_by_service
        if traffic_intensity_by_service is not None
        else state.traffic_intensity_by_service
    )

    def get_rho(service_id: int) -> float:
        if service_id in rho_map:
            return rho_map[service_id]
        svc_cfg = state.services.get(service_id)
        if svc_cfg and svc_cfg.mean_service_minutes > 0.0:
            active_count = sum(
                1 for other in state.counters.values()
                if other.is_active and other.assigned_service_id == service_id
            )
            mu = active_count * (60.0 / svc_cfg.mean_service_minutes)
            q_len = state.queues[service_id].length() if service_id in state.queues else 0
            return q_len / mu if mu > 0.0 else float(q_len)
        return float(state.queues[service_id].length()) if service_id in state.queues else 0.0

    # a. Target service = service with longest current waiting queue
    sorted_queues = sorted(
        state.queues.items(),
        key=lambda item: (-item[1].length(), item[0]),
    )

    if not sorted_queues or sorted_queues[0][1].length() <= 0:
        return None

    target_s_id = sorted_queues[0][0]

    # b. If target service is STABLE, return no action
    target_rho = get_rho(target_s_id)
    if is_stable(target_rho):
        return None

    # c. Valid donors: active, eligible, not assigned to target, and donor's source service
    # must retain at least 1 active assigned counter after reassignment.
    def source_keeps_active_counter(c: SimCounter) -> bool:
        active_source_count = sum(
            1 for other in state.counters.values()
            if other.is_active and other.assigned_service_id == c.assigned_service_id
        )
        return active_source_count >= 2

    valid_donors = [
        c for c in state.counters.values()
        if c.is_active
        and c.can_serve(target_s_id)
        and c.assigned_service_id != target_s_id
        and source_keeps_active_counter(c)
    ]

    if not valid_donors:
        return None

    # d. Deterministic ranking:
    # 1. lowest source service rho
    # 2. prefer idle donor
    # 3. lowest counter_id tie-break
    valid_donors.sort(
        key=lambda c: (
            get_rho(c.assigned_service_id),
            1 if c.is_busy else 0,
            c.counter_id,
        )
    )

    chosen = valid_donors[0]
    cand_suffix = f"_{int(duration_min)}" if duration_min is not None else ""
    return InterventionCandidate(
        candidate_id=f"NAIVE_LONGEST_QUEUE{cand_suffix}",
        counter_id=chosen.counter_id,
        target_service_id=target_s_id,
        source_service_id=chosen.assigned_service_id,
        effective_time_min=0.0,
        duration_min=duration_min,
        switching_delay_min=switching_delay_min,
        description=(
            f"Naive longest-queue reassignment: Counter {chosen.counter_id} ({chosen.name}) "
            f"reassigned to service {target_s_id} for {duration_min:.0f}m"
        ),
    )


def generate_candidate_interventions(
    state: SimulationState,
    max_candidates: int = 10,
    durations: Optional[Sequence[float]] = None,
    switching_delay_min: float = 3.0,
) -> List[InterventionCandidate]:
    """
    Generates a bounded, explainable set of valid counter reassignment candidates.
    Constraints:
    - Counter must be active.
    - Target service must be eligible for that counter.
    - Target service must be different from the counter's current assignment.
    - Target service must have waiting citizens (queue backlog).
    - Ranked by queue deficit and counter availability.
    - Supports multiple candidate durations (e.g. 15m, 30m) when specified.
    """
    candidates: List[InterventionCandidate] = []
    seen_combinations = set()

    # Identify potential target services with waiting citizens
    congested_services = [
        (s_id, q.length()) for s_id, q in state.queues.items() if q.length() > 0
    ]
    congested_services.sort(key=lambda x: -x[1])

    if not congested_services:
        return []

    for target_s_id, target_len in congested_services:
        # Find active counters that can legally serve target_s_id
        eligible_counters = [
            c for c in state.counters.values()
            if c.is_active and c.can_serve(target_s_id) and c.assigned_service_id != target_s_id
        ]

        for counter in eligible_counters:
            combo = (counter.counter_id, target_s_id)
            if combo in seen_combinations:
                continue
            seen_combinations.add(combo)

            source_len = state.queues[counter.assigned_service_id].length()
            imbalance = target_len - source_len

            if durations is not None and len(durations) > 0:
                for dur in durations:
                    candidates.append(
                        InterventionCandidate(
                            candidate_id=f"CAND_C{counter.counter_id}_TO_S{target_s_id}_D{int(dur)}",
                            counter_id=counter.counter_id,
                            target_service_id=target_s_id,
                            source_service_id=counter.assigned_service_id,
                            effective_time_min=0.0,
                            duration_min=float(dur),
                            switching_delay_min=switching_delay_min,
                            description=(
                                f"Reassign Counter {counter.counter_id} ({counter.name}) from "
                                f"Service {counter.assigned_service_id} (queue={source_len}) to "
                                f"Service {target_s_id} (queue={target_len}) for {dur:.0f}m"
                            ),
                        )
                    )
            else:
                candidates.append(
                    InterventionCandidate(
                        candidate_id=f"CAND_C{counter.counter_id}_TO_S{target_s_id}",
                        counter_id=counter.counter_id,
                        target_service_id=target_s_id,
                        source_service_id=counter.assigned_service_id,
                        effective_time_min=0.0,
                        duration_min=None,
                        switching_delay_min=switching_delay_min,
                        description=(
                            f"Reassign Counter {counter.counter_id} ({counter.name}) from "
                            f"Service {counter.assigned_service_id} (queue={source_len}) to "
                            f"Service {target_s_id} (queue={target_len})"
                        ),
                    )
                )

    # Rank candidates: idle counters first, highest queue imbalance, deterministic tie-breaking
    candidates.sort(
        key=lambda c: (
            1 if state.counters[c.counter_id].is_busy else 0,
            -(state.queues[c.target_service_id].length() - state.queues[state.counters[c.counter_id].assigned_service_id].length()),
            c.counter_id,
            c.target_service_id,
            c.duration_min or 0.0,
        )
    )

    return candidates[:max_candidates]
