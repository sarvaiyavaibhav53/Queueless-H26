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

from typing import List, Optional, Sequence

from app.interventions.models import InterventionCandidate
from app.simulation.state import SimulationState


def build_naive_intervention(state: SimulationState) -> Optional[InterventionCandidate]:
    """
    Constructs the NAIVE_LONGEST_QUEUE policy:
    1. Inspects current waiting queue lengths across all services.
    2. Identifies the service with the longest waiting queue.
    3. Finds an eligible active counter not already assigned to that service.
    4. Prefers an idle counter.
    5. Returns candidate or None if no valid reassignment exists (falls back to NO_INTERVENTION).
    """
    # Sort services by queue length descending, tie-break by service_id ascending
    sorted_services = sorted(
        state.queues.items(),
        key=lambda item: (-item[1].length(), item[0]),
    )

    for target_s_id, target_queue in sorted_services:
        if target_queue.length() <= 0:
            # All remaining queues have length 0; no reassignment justified
            break

        # Find eligible active counters not currently assigned to target_s_id
        eligible_counters = [
            c for c in state.counters.values()
            if c.is_active and c.can_serve(target_s_id) and c.assigned_service_id != target_s_id
        ]

        if not eligible_counters:
            continue

        # Preference: Idle counters first, then sort by source queue length ascending, then counter_id
        eligible_counters.sort(
            key=lambda c: (
                1 if c.is_busy else 0,
                state.queues[c.assigned_service_id].length(),
                c.counter_id,
            )
        )

        chosen = eligible_counters[0]
        return InterventionCandidate(
            candidate_id="NAIVE_LONGEST_QUEUE",
            counter_id=chosen.counter_id,
            target_service_id=target_s_id,
            source_service_id=chosen.assigned_service_id,
            effective_time_min=0.0,
            duration_min=None,
            switching_delay_min=3.0,
            description=(
                f"Naive longest-queue reassignment: Counter {chosen.counter_id} ({chosen.name}) "
                f"reassigned to service {target_s_id}"
            ),
        )

    return None


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
