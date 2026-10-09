"""
PRAVAAH — Intervention Evaluation Schemas & DTOs.

Defines Pydantic request and response models for evaluating candidate interventions
via the FastAPI API boundary.
Guarantees:
- Strict Pydantic input validation.
- Complete decoupling: does not expose internal Python or SQLAlchemy objects.
- Explicit simulation-only markers (applied=False, is_simulation_only=True).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class InterventionEvaluationRequest(BaseModel):
    """
    Request payload to evaluate candidate counter interventions for a specific office.
    """
    office_id: int = Field(
        ...,
        gt=0,
        description="Unique database identifier of the target office to evaluate.",
    )
    horizon_minutes: Optional[float] = Field(
        default=60.0,
        gt=0.0,
        le=480.0,
        description="Simulation horizon in minutes (must be > 0 and <= 480). Default: 60.0.",
    )
    replications: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of stochastic Monte Carlo replications (1 to 50). Default: 5.",
    )
    base_seed: Optional[int] = Field(
        default=42,
        ge=0,
        description="Base random seed for reproducible Common Random Numbers. Default: 42.",
    )
    p90_thresholds: Optional[Dict[int, float]] = Field(
        default=None,
        description="Optional mapping of service_id to maximum allowable P90 waiting minutes.",
    )
    max_candidates: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of candidate interventions to evaluate (1 to 50). Default: 10.",
    )
    arrival_rates_per_hour: Optional[Dict[int, float]] = Field(
        default=None,
        description="Optional Poisson arrival rate per hour per service (service_id -> rate).",
    )

    @field_validator("p90_thresholds")
    @classmethod
    def validate_p90_thresholds(
        cls, v: Optional[Dict[int, float]]
    ) -> Optional[Dict[int, float]]:
        if v is not None:
            for s_id, thresh in v.items():
                if s_id <= 0:
                    raise ValueError(f"Service ID must be positive (got {s_id})")
                if thresh < 0.0:
                    raise ValueError(f"P90 threshold must be non-negative (got {thresh})")
        return v

    @field_validator("arrival_rates_per_hour")
    @classmethod
    def validate_arrival_rates(
        cls, v: Optional[Dict[int, float]]
    ) -> Optional[Dict[int, float]]:
        if v is not None:
            for s_id, rate in v.items():
                if s_id <= 0:
                    raise ValueError(f"Service ID must be positive (got {s_id})")
                if rate < 0.0:
                    raise ValueError(f"Arrival rate must be non-negative (got {rate})")
        return v


class InterventionCandidateDTO(BaseModel):
    """
    Details of an operational counter reassignment intervention candidate.
    """
    candidate_id: str
    counter_id: int
    target_service_id: int
    source_service_id: Optional[int] = None
    effective_time_min: float = 0.0
    duration_min: Optional[float] = None
    switching_delay_min: float = 3.0
    description: str = ""


class PolicyEvaluationDTO(BaseModel):
    """
    Simulated performance metrics for a specific operational policy.
    """
    policy_type: str
    candidate: Optional[InterventionCandidateDTO] = None
    is_feasible: bool
    rejection_reason: Optional[str] = None
    mean_total_waiting_person_minutes: float
    mean_served_waiting_person_minutes: float
    mean_unserved_waiting_person_minutes: float
    mean_customers_served: float
    p90_wait_by_service: Dict[int, float]
    threshold_by_service: Dict[int, Optional[float]]
    replication_count: int


class InterventionEvaluationResponse(BaseModel):
    """
    Comprehensive structured response for intervention evaluation.
    Clearly marks recommendations as read-only and unapplied.
    """
    office_id: int
    snapshot_time_utc: datetime
    selected_policy: str
    selected_candidate: Optional[InterventionCandidateDTO] = None
    is_feasible: bool
    total_waiting_person_minutes: float
    improvement_vs_baseline: float
    pct_improvement_vs_baseline: float
    improvement_vs_naive: float
    pct_improvement_vs_naive: float
    per_service_p90: Dict[int, float]
    configured_p90_thresholds: Dict[int, Optional[float]]
    baseline_result: PolicyEvaluationDTO
    naive_result: PolicyEvaluationDTO
    optimized_result: Optional[PolicyEvaluationDTO] = None
    candidate_evaluations: List[PolicyEvaluationDTO] = Field(default_factory=list)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    replication_count: int
    horizon_minutes: float
    base_seed: int

    # Safety and persistence indicators
    applied: bool = False
    is_simulation_only: bool = True
    status_message: str = (
        "Intervention evaluated successfully. "
        "Recommendation generated for advisory preview only; not applied to live operations."
    )


class ApplyInterventionRequest(BaseModel):
    office_id: int
    counter_id: int
    target_service_id: int


class ApplyInterventionResponse(BaseModel):
    applied: bool = True
    office_id: int
    counter_id: int
    previous_service_id: int
    target_service_id: int
    reassigned_at_utc: datetime
    switching_delay_note: str = (
        "Operational database reassignment is immediate. "
        "The 3.0-minute switching delay is modeled strictly in discrete-event simulation look-ahead."
    )


class RestoreInterventionRequest(BaseModel):
    office_id: int
    counter_id: int


class RestoreInterventionResponse(BaseModel):
    restored: bool = True
    office_id: int
    counter_id: int
    previous_service_id: int
    restored_service_id: int
    restored_at_utc: datetime
    message: str

