"""
PRAVAAH — Simulation Replication Metrics & P90 Percentile Calculation.

Pure Python percentile calculation and replication aggregation.
Guarantees:
- Deterministic linear interpolation percentile calculation (NIST / NumPy linear standard).
- Handles N=0, N=1, and small sample sizes safely without producing NaN or raising exceptions.
- Enforces strict P90 threshold feasibility checks across replications.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from app.interventions.models import (
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)
from app.simulation.results import SimulationResult


def calculate_percentile(values: Sequence[float], p: float = 90.0) -> float:
    """
    Computes deterministic p-th percentile (0.0 <= p <= 100.0) using linear interpolation.
    
    Behavior on edge cases:
    - N == 0: Returns 0.0 (zero observed waiting congestion).
    - N == 1: Returns float(values[0]).
    - N >= 2: Linear interpolation between adjacent rank values.
    """
    if not values:
        return 0.0

    if len(values) == 1:
        return float(values[0])

    if p < 0.0 or p > 100.0:
        raise ValueError(f"Percentile p must be in [0.0, 100.0] (got {p})")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Continuous index k in [0, n - 1]
    k = (n - 1) * (p / 100.0)
    floor_idx = math.floor(k)
    ceil_idx = math.ceil(k)

    if floor_idx == ceil_idx:
        return float(sorted_vals[int(k)])

    weight_ceil = k - floor_idx
    weight_floor = ceil_idx - k
    interpolated = (sorted_vals[floor_idx] * weight_floor) + (sorted_vals[ceil_idx] * weight_ceil)
    return float(interpolated)


def aggregate_replications(
    policy_type: PolicyType,
    candidate: Optional[InterventionCandidate],
    results: List[SimulationResult],
    p90_thresholds: Dict[int, float],
    service_ids: Sequence[int],
) -> PolicyEvaluation:
    """
    Aggregates multiple simulation replication results for a candidate policy,
    evaluates P90 waiting times per service, and assesses feasibility constraints.
    """
    if not results:
        raise ValueError("Cannot aggregate empty simulation results list")

    num_reps = len(results)
    total_waits = tuple(r.total_waiting_person_minutes for r in results)
    mean_total_wait = sum(total_waits) / num_reps
    mean_served_wait = sum(r.served_waiting_person_minutes for r in results) / num_reps
    mean_unserved_wait = sum(r.unserved_waiting_person_minutes for r in results) / num_reps
    mean_served_count = sum(r.total_customers_served for r in results) / num_reps

    # Aggregate individual customer waiting times per service across all replications
    waiting_times_by_service: Dict[int, List[float]] = {s_id: [] for s_id in service_ids}
    for res in results:
        for s_id in service_ids:
            waits = res.service_waiting_times.get(s_id, ())
            waiting_times_by_service[s_id].extend(waits)

    # Compute P90 wait time per service
    p90_by_service: Dict[int, float] = {}
    for s_id in service_ids:
        p90_by_service[s_id] = calculate_percentile(waiting_times_by_service[s_id], p=90.0)

    # Feasibility check against configured P90 thresholds
    is_feasible = True
    rejection_reason: Optional[str] = None
    threshold_by_service: Dict[int, Optional[float]] = {}

    for s_id in service_ids:
        thresh = p90_thresholds.get(s_id)
        threshold_by_service[s_id] = thresh
        if thresh is not None:
            observed_p90 = p90_by_service[s_id]
            if observed_p90 > thresh + 1e-9:
                is_feasible = False
                cand_label = candidate.candidate_id if candidate else policy_type.value
                rejection_reason = (
                    f"Candidate {cand_label} violated Service {s_id} P90 threshold: "
                    f"observed {observed_p90:.2f}m > allowed {thresh:.2f}m"
                )

    return PolicyEvaluation(
        policy_type=policy_type,
        candidate=candidate,
        is_feasible=is_feasible,
        rejection_reason=rejection_reason,
        mean_total_waiting_person_minutes=mean_total_wait,
        mean_served_waiting_person_minutes=mean_served_wait,
        mean_unserved_waiting_person_minutes=mean_unserved_wait,
        mean_customers_served=mean_served_count,
        p90_wait_by_service=p90_by_service,
        threshold_by_service=threshold_by_service,
        replication_count=num_reps,
        replication_total_waits=total_waits,
    )
