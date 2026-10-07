"""
PRAVAAH — Candidate Intervention Domain Models & Typed Evaluator Data Structures.

Pure Python typed definitions for intervention candidates, evaluator configuration,
policy evaluations, and decision results.
"""

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple


@unique
class PolicyType(str, Enum):
    BASELINE = "BASELINE"
    NAIVE = "NAIVE"
    OPTIMIZED = "OPTIMIZED"


@dataclass(frozen=True)
class InterventionCandidate:
    """
    Proposed operational intervention to reassign a physical counter.
    """
    candidate_id: str
    counter_id: int
    target_service_id: int
    source_service_id: Optional[int] = None
    effective_time_min: float = 0.0
    duration_min: Optional[float] = None
    switching_delay_min: float = 3.0
    description: str = ""


@dataclass(frozen=True)
class EvaluatorConfig:
    """
    Configuration parameters controlling candidate evaluation and P90 constraints.
    """
    horizon_minutes: float = 60.0
    replications: int = 5
    base_seed: int = 42
    p90_thresholds: Dict[int, float] = field(default_factory=dict)  # service_id -> P90 threshold (min)
    max_candidates: int = 10
    arrival_rates_per_hour: Optional[Dict[int, float]] = None
    candidate_durations: Optional[Sequence[float]] = None
    switching_delay_min: float = 3.0
    availability_events: Sequence[Any] = field(default_factory=tuple)
    arrival_process_factory: Optional[Any] = None
    service_process_factory: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.horizon_minutes <= 0.0:
            raise ValueError(f"horizon_minutes must be > 0.0 (got {self.horizon_minutes})")
        if self.replications < 1:
            raise ValueError(f"replications must be >= 1 (got {self.replications})")
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1 (got {self.max_candidates})")
        if self.switching_delay_min < 0.0:
            raise ValueError(f"switching_delay_min must be >= 0.0 (got {self.switching_delay_min})")
        for s_id, thresh in self.p90_thresholds.items():
            if thresh < 0.0:
                raise ValueError(f"P90 threshold for service {s_id} must be >= 0.0 (got {thresh})")


@dataclass(frozen=True)
class PolicyEvaluation:
    """
    Aggregated performance of a single candidate policy across multiple replications.
    """
    policy_type: PolicyType
    candidate: Optional[InterventionCandidate]
    is_feasible: bool
    rejection_reason: Optional[str]
    mean_total_waiting_person_minutes: float
    mean_served_waiting_person_minutes: float
    mean_unserved_waiting_person_minutes: float
    mean_customers_served: float
    p90_wait_by_service: Dict[int, float]
    threshold_by_service: Dict[int, Optional[float]]
    replication_count: int
    replication_total_waits: Tuple[float, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_type": self.policy_type.value,
            "candidate_id": self.candidate.candidate_id if self.candidate else None,
            "counter_id": self.candidate.counter_id if self.candidate else None,
            "target_service_id": self.candidate.target_service_id if self.candidate else None,
            "is_feasible": self.is_feasible,
            "rejection_reason": self.rejection_reason,
            "mean_total_waiting_person_minutes": round(self.mean_total_waiting_person_minutes, 4),
            "mean_served_waiting_person_minutes": round(self.mean_served_waiting_person_minutes, 4),
            "mean_unserved_waiting_person_minutes": round(self.mean_unserved_waiting_person_minutes, 4),
            "mean_customers_served": round(self.mean_customers_served, 2),
            "p90_wait_by_service": {k: round(v, 4) for k, v in self.p90_wait_by_service.items()},
            "threshold_by_service": self.threshold_by_service,
            "replication_count": self.replication_count,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """
    Final decision output comparing Baseline, Naive, and PRAVAAH Optimized policies.
    """
    selected_policy: PolicyType
    selected_candidate: Optional[InterventionCandidate]
    is_feasible: bool
    total_waiting_person_minutes: float
    improvement_vs_baseline: float
    pct_improvement_vs_baseline: float
    improvement_vs_naive: float
    pct_improvement_vs_naive: float
    baseline_evaluation: PolicyEvaluation
    naive_evaluation: PolicyEvaluation
    candidate_evaluations: List[PolicyEvaluation]
    explanation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_policy": self.selected_policy.value,
            "selected_candidate": (
                self.selected_candidate.candidate_id if self.selected_candidate else None
            ),
            "is_feasible": self.is_feasible,
            "total_waiting_person_minutes": round(self.total_waiting_person_minutes, 4),
            "improvement_vs_baseline": round(self.improvement_vs_baseline, 4),
            "pct_improvement_vs_baseline": round(self.pct_improvement_vs_baseline, 2),
            "improvement_vs_naive": round(self.improvement_vs_naive, 4),
            "pct_improvement_vs_naive": round(self.pct_improvement_vs_naive, 2),
            "baseline": self.baseline_evaluation.to_dict(),
            "naive": self.naive_evaluation.to_dict(),
            "candidate_evaluations": [c.to_dict() for c in self.candidate_evaluations],
            "explanation": self.explanation,
        }
