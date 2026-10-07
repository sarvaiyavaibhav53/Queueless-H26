"""
PRAVAAH — Candidate Intervention Evaluation & Decision Engine Package.

Provides candidate generation, naive baseline formulation, replication simulation,
P90 constraint evaluation, and optimal policy recommendation.
"""

from app.interventions.candidates import (
    build_naive_intervention,
    generate_candidate_interventions,
)
from app.interventions.evaluator import (
    InterventionEvaluator,
)
from app.interventions.metrics import (
    aggregate_replications,
    calculate_percentile,
)
from app.interventions.models import (
    EvaluatorConfig,
    EvaluationResult,
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)

__all__ = [
    "InterventionEvaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "PolicyEvaluation",
    "PolicyType",
    "InterventionCandidate",
    "build_naive_intervention",
    "generate_candidate_interventions",
    "calculate_percentile",
    "aggregate_replications",
]
