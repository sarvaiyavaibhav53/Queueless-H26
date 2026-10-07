"""
PRAVAAH — Intervention Evaluator API Endpoint.

Provides secure, read-only endpoint POST /api/v1/interventions/evaluate for admins
to preview simulation-driven counter intervention recommendations.

Architecture Guarantees:
- Admin-only access enforced via RBAC (require_admin).
- Office existence and object-level scoping validated before simulation.
- Explicit timezone-aware UTC point-in-time timestamp generated at API boundary.
- Zero database state mutation (no interventions applied, no recommendations persisted).
- Decoupled pure domain engine: simulation and evaluator remain isolated from HTTP/DB layers.
- Safe error handling: no raw stack traces, SQL errors, filesystem paths, or internal details exposed.
"""

from datetime import datetime, timezone
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.database import get_db
from app.interventions.evaluator import InterventionEvaluator
from app.interventions.models import (
    EvaluatorConfig,
    EvaluationResult,
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)
from app.models.office import Office
from app.models.user import User
from app.schemas.intervention import (
    InterventionCandidateDTO,
    InterventionEvaluationRequest,
    InterventionEvaluationResponse,
    PolicyEvaluationDTO,
)
from app.services.snapshot_adapter import (
    SnapshotAdapterError,
    build_simulation_snapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/interventions", tags=["interventions"])


def _map_candidate_to_dto(
    candidate: Optional[InterventionCandidate],
) -> Optional[InterventionCandidateDTO]:
    """Maps internal domain InterventionCandidate to safe API response DTO."""
    if candidate is None:
        return None
    return InterventionCandidateDTO(
        candidate_id=candidate.candidate_id,
        counter_id=candidate.counter_id,
        target_service_id=candidate.target_service_id,
        source_service_id=candidate.source_service_id,
        effective_time_min=candidate.effective_time_min,
        duration_min=candidate.duration_min,
        switching_delay_min=candidate.switching_delay_min,
        description=candidate.description,
    )


def _map_policy_evaluation_to_dto(pe: PolicyEvaluation) -> PolicyEvaluationDTO:
    """Maps internal PolicyEvaluation to safe API response DTO."""
    return PolicyEvaluationDTO(
        policy_type=pe.policy_type.value if hasattr(pe.policy_type, "value") else str(pe.policy_type),
        candidate=_map_candidate_to_dto(pe.candidate),
        is_feasible=pe.is_feasible,
        rejection_reason=pe.rejection_reason,
        mean_total_waiting_person_minutes=round(pe.mean_total_waiting_person_minutes, 4),
        mean_served_waiting_person_minutes=round(pe.mean_served_waiting_person_minutes, 4),
        mean_unserved_waiting_person_minutes=round(pe.mean_unserved_waiting_person_minutes, 4),
        mean_customers_served=round(pe.mean_customers_served, 2),
        p90_wait_by_service={k: round(v, 4) for k, v in pe.p90_wait_by_service.items()},
        threshold_by_service=pe.threshold_by_service,
        replication_count=pe.replication_count,
    )


@router.post(
    "/evaluate",
    response_model=InterventionEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate operational counter interventions",
    description=(
        "Simulates and evaluates operational counter reassignments (Baseline vs Naive vs Optimized) "
        "for an authorized office. Returns comparative metrics without persisting or applying changes."
    ),
)
def evaluate_interventions(
    request: InterventionEvaluationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
) -> InterventionEvaluationResponse:
    """
    Evaluates candidate interventions for the specified office.

    Enforces:
    - Admin-only access.
    - Object-level validation: target office must exist.
    - Explicit UTC timestamp passed to simulation snapshot builder.
    - Read-only simulation execution: no mutations to database state.
    """
    # 1. Enforce object-level office validation
    office = db.query(Office).filter(Office.id == request.office_id).first()
    if not office:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Office with ID {request.office_id} not found",
        )

    # 2. Acquire current point-in-time UTC timestamp at API boundary
    snapshot_time_utc = datetime.now(timezone.utc)

    # 3. Build pure simulation snapshot from database state
    try:
        snapshot = build_simulation_snapshot(
            db=db,
            office_id=request.office_id,
            snapshot_time_utc=snapshot_time_utc,
        )
    except SnapshotAdapterError as sae:
        logger.warning(
            "SnapshotAdapterError for office %d: %s", request.office_id, str(sae)
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Simulation snapshot generation failed: {str(sae)}",
        )
    except Exception as exc:
        logger.error(
            "Unexpected error building snapshot for office %d: %s",
            request.office_id,
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the simulation snapshot.",
        )

    # 4. Instantiate evaluator configuration and execute simulation evaluation
    try:
        evaluator_config = EvaluatorConfig(
            horizon_minutes=request.horizon_minutes if request.horizon_minutes is not None else 60.0,
            replications=request.replications if request.replications is not None else 5,
            base_seed=request.base_seed if request.base_seed is not None else 42,
            p90_thresholds=request.p90_thresholds or {},
            max_candidates=request.max_candidates if request.max_candidates is not None else 10,
            arrival_rates_per_hour=request.arrival_rates_per_hour,
        )
        evaluator = InterventionEvaluator(config=evaluator_config)
        evaluation_result: EvaluationResult = evaluator.evaluate(snapshot)
    except ValueError as ve:
        logger.warning(
            "Parameter validation error during evaluation for office %d: %s",
            request.office_id,
            str(ve),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid evaluation configuration: {str(ve)}",
        )
    except Exception as exc:
        logger.error(
            "Unexpected error during intervention evaluation for office %d: %s",
            request.office_id,
            str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during intervention evaluation.",
        )

    # 5. Map pure domain EvaluationResult into API response DTO
    baseline_dto = _map_policy_evaluation_to_dto(evaluation_result.baseline_evaluation)
    naive_dto = _map_policy_evaluation_to_dto(evaluation_result.naive_evaluation)
    candidate_dtos = [
        _map_policy_evaluation_to_dto(c) for c in evaluation_result.candidate_evaluations
    ]

    # Resolve optimized evaluation DTO
    optimized_dto: Optional[PolicyEvaluationDTO] = None
    if (
        evaluation_result.selected_policy == PolicyType.OPTIMIZED
        and evaluation_result.selected_candidate
    ):
        for c_dto in candidate_dtos:
            if (
                c_dto.candidate
                and c_dto.candidate.candidate_id
                == evaluation_result.selected_candidate.candidate_id
            ):
                optimized_dto = c_dto
                break
    elif candidate_dtos:
        optimized_dto = candidate_dtos[0]

    # Determine per-service P90 representation for selected policy
    if evaluation_result.selected_policy == PolicyType.BASELINE:
        selected_p90 = baseline_dto.p90_wait_by_service
    elif evaluation_result.selected_policy == PolicyType.NAIVE:
        selected_p90 = naive_dto.p90_wait_by_service
    elif optimized_dto is not None:
        selected_p90 = optimized_dto.p90_wait_by_service
    else:
        selected_p90 = baseline_dto.p90_wait_by_service

    return InterventionEvaluationResponse(
        office_id=request.office_id,
        snapshot_time_utc=snapshot_time_utc,
        selected_policy=(
            evaluation_result.selected_policy.value
            if hasattr(evaluation_result.selected_policy, "value")
            else str(evaluation_result.selected_policy)
        ),
        selected_candidate=_map_candidate_to_dto(evaluation_result.selected_candidate),
        is_feasible=evaluation_result.is_feasible,
        total_waiting_person_minutes=round(evaluation_result.total_waiting_person_minutes, 4),
        improvement_vs_baseline=round(evaluation_result.improvement_vs_baseline, 4),
        pct_improvement_vs_baseline=round(evaluation_result.pct_improvement_vs_baseline, 2),
        improvement_vs_naive=round(evaluation_result.improvement_vs_naive, 4),
        pct_improvement_vs_naive=round(evaluation_result.pct_improvement_vs_naive, 2),
        per_service_p90=selected_p90,
        configured_p90_thresholds=baseline_dto.threshold_by_service,
        baseline_result=baseline_dto,
        naive_result=naive_dto,
        optimized_result=optimized_dto,
        candidate_evaluations=candidate_dtos,
        explanation=evaluation_result.explanation,
        replication_count=evaluator_config.replications,
        horizon_minutes=evaluator_config.horizon_minutes,
        base_seed=evaluator_config.base_seed,
        applied=False,
        is_simulation_only=True,
    )
