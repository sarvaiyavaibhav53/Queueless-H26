"""
PRAVAAH — Intervention Policy Helpers.

Bridges candidate interventions into DiscreteEventEngine intervention schedules.
"""

from typing import List, Optional

from app.interventions.models import InterventionCandidate
from app.simulation.engine import CounterReassignIntervention
from app.simulation.state import SimulationState


def candidate_to_engine_interventions(
    candidate: Optional[InterventionCandidate],
) -> List[CounterReassignIntervention]:
    """
    Translates an InterventionCandidate into a list of engine CounterReassignIntervention hooks.
    """
    if candidate is None:
        return []

    return [
        CounterReassignIntervention(
            time_min=candidate.effective_time_min,
            counter_id=candidate.counter_id,
            new_service_id=candidate.target_service_id,
            duration_min=candidate.duration_min,
            switching_delay_min=candidate.switching_delay_min,
            original_service_id=candidate.source_service_id,
        )
    ]


def validate_candidate_against_state(
    candidate: InterventionCandidate,
    state: SimulationState,
) -> None:
    """
    Ensures candidate satisfies all domain safety invariants:
    - Counter exists and is active.
    - Target service exists.
    - Counter is eligible for target service.
    - Counter is not already assigned to target service.
    - Duration (if specified) is positive.
    - Switching delay is non-negative.
    - Source service (if specified) matches counter assignment.
    """
    if candidate.counter_id not in state.counters:
        raise ValueError(f"Counter {candidate.counter_id} does not exist in simulation state")

    counter = state.counters[candidate.counter_id]
    if not counter.is_active:
        raise ValueError(f"Counter {counter.counter_id} is UNAVAILABLE and cannot be reassigned")

    if candidate.target_service_id not in state.services:
        raise ValueError(f"Target service {candidate.target_service_id} does not exist in state")

    if not counter.can_serve(candidate.target_service_id):
        raise ValueError(
            f"Counter {counter.counter_id} is not eligible for service {candidate.target_service_id} "
            f"(eligible services: {set(counter.eligible_service_ids)})"
        )

    if counter.assigned_service_id == candidate.target_service_id:
        raise ValueError(
            f"Counter {counter.counter_id} is already assigned to service {candidate.target_service_id}"
        )

    if candidate.source_service_id is not None and candidate.source_service_id != counter.assigned_service_id:
        raise ValueError(
            f"Candidate source service {candidate.source_service_id} does not match counter "
            f"{counter.counter_id} current assigned service {counter.assigned_service_id}"
        )

    if candidate.duration_min is not None and candidate.duration_min <= 0.0:
        raise ValueError(f"Intervention duration must be > 0.0 minutes (got {candidate.duration_min})")

    if candidate.switching_delay_min < 0.0:
        raise ValueError(f"Switching delay must be >= 0.0 minutes (got {candidate.switching_delay_min})")
