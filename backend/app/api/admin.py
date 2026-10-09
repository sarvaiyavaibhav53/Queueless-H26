"""
PRAVAAH — Admin Command-Center API Endpoints.

Provides authenticated, office-scoped administrative operations:
- POST /api/v1/admin/offices/{office_id}/scenarios/load: Load frozen simulation scenario
- GET  /api/v1/admin/scenarios: List available frozen scenarios
- GET  /api/v1/admin/offices/{office_id}/summary: Real-time operational command-center snapshot
"""

import math
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.core.config import settings
from app.db.database import get_db
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.models.user import User
from app.schemas.admin import (
    ScenarioLoadRequest,
    ScenarioLoadResponse,
    AvailableScenariosResponse,
    AdminOperationsSummaryResponse,
    ServiceSummaryDTO,
    CounterSummaryDTO,
)
from app.services.scenario_loader_service import (
    load_scenario_into_db,
    get_available_scenarios,
    ScenarioLoaderError,
)
from app.services.snapshot_adapter import (
    build_simulation_snapshot,
    SnapshotAdapterError,
)
from shared.features import compute_raw_features
from shared.stability import classify_stability
from app.ws_manager import ws_manager

logger = logging.getLogger("pravaah.api.admin")


def compute_time_to_congestion_fluid(
    queue_length: int,
    arrival_rate_per_hour: float,
    service_capacity_per_hour: float,
    congestion_threshold: int = 15,
) -> Tuple[Optional[float], str]:
    """
    Computes fluid approximation of time to reach queue congestion threshold.

    Units:
      - queue_length: count of waiting citizens (dimensionless)
      - arrival_rate_per_hour: lambda (citizens/hour)
      - service_capacity_per_hour: mu (citizens/hour)
      - congestion_threshold: critical backlog threshold (default: 15 tokens)

    Assumptions & Semantics:
      - Fluid model assuming constant arrival rate lambda and service capacity mu.
      - Both lambda and mu are expressed in citizens/hour.
      - Net backlog accumulation rate = lambda - mu (citizens/hour) = (lambda - mu) / 60.0 (citizens/min).
      - If queue_length >= congestion_threshold: returns 0.0 (already congested).
      - If lambda <= mu: returns None (queue is stable or draining; will not congest under current rates).
      - If lambda > mu and queue_length < congestion_threshold:
          remaining_tokens = congestion_threshold - queue_length
          ttc_minutes = remaining_tokens / ((lambda - mu) / 60.0)
      - Never returns negative minutes, NaN, or infinity.
    """
    method = "fluid_approximation"
    if congestion_threshold <= 0:
        return (None, method)
    if queue_length < 0 or arrival_rate_per_hour < 0 or service_capacity_per_hour < 0:
        return (None, method)

    if queue_length >= congestion_threshold:
        return (0.0, method)

    net_growth_per_hour = arrival_rate_per_hour - service_capacity_per_hour
    if net_growth_per_hour <= 0.0:
        return (None, method)

    net_growth_per_minute = net_growth_per_hour / 60.0
    tokens_remaining = congestion_threshold - queue_length
    ttc_minutes = tokens_remaining / net_growth_per_minute

    if not math.isfinite(ttc_minutes) or ttc_minutes < 0.0:
        return (None, method)

    return (round(ttc_minutes, 1), method)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get(
    "/scenarios",
    response_model=AvailableScenariosResponse,
    summary="List available simulation scenarios",
)
def list_scenarios(current_user: User = Depends(require_admin)):
    """
    Returns metadata for all frozen simulation scenarios available to load.
    """
    scenarios_dict = get_available_scenarios()
    scenarios_list = [
        {"scenario_id": sid, "scenario_name": sname}
        for sid, sname in sorted(scenarios_dict.items())
    ]
    return AvailableScenariosResponse(scenarios=scenarios_list)


@router.post(
    "/offices/{office_id}/scenarios/load",
    response_model=ScenarioLoadResponse,
    summary="Load and reset deterministic scenario into office database",
)
def load_office_scenario(
    office_id: int,
    request: ScenarioLoadRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Atomically resets and loads the specified simulation scenario into the database
    for the given office. Strictly office-scoped.
    Emits invalidation events for affected services post-commit.
    """
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Office with ID {office_id} not found",
        )

    try:
        result = load_scenario_into_db(
            db=db,
            office_id=office_id,
            scenario_id=request.scenario_id,
            now=request.now,
        )
    except ScenarioLoaderError as sle:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(sle),
        )
    except Exception as exc:
        logger.error(f"Failed to load scenario: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load scenario due to internal database error",
        )

    # Post-commit: Broadcast invalidation hints for all affected services
    for s_id in result.affected_service_ids:
        background_tasks.add_task(
            ws_manager.broadcast_invalidation,
            event_type="queue.updated",
            office_id=office_id,
            service_id=s_id,
        )

    return ScenarioLoadResponse(
        scenario_id=result.scenario_id,
        scenario_name=result.scenario_name,
        office_id=result.office_id,
        office_code=result.office_code,
        affected_service_ids=result.affected_service_ids,
        counters_loaded=result.counters_loaded,
        waiting_tokens_loaded=result.waiting_tokens_loaded,
        busy_counters_loaded=result.busy_counters_loaded,
        loaded_at_utc=result.loaded_at_utc,
    )


@router.get(
    "/offices/{office_id}/summary",
    response_model=AdminOperationsSummaryResponse,
    summary="Get real-time command center operational summary for an office",
)
def get_office_operations_summary(
    office_id: int,
    congestion_threshold: Optional[int] = Query(None, description="Queue backlog threshold for fluid time-to-congestion (defaults to configured setting)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Provides real-time operational metrics for all services and physical counters
    in an office. Reuses the authoritative simulation snapshot adapter and shared
    stability classifiers. Strictly read-only with zero database mutations.
    """
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Office with ID {office_id} not found",
        )

    now_utc = datetime.now(timezone.utc)

    # 1. Build canonical snapshot
    try:
        snapshot = build_simulation_snapshot(
            db=db,
            office_id=office_id,
            snapshot_time_utc=now_utc,
        )
    except SnapshotAdapterError as sae:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to build operations snapshot: {str(sae)}",
        )

    # 2. Compile per-service metrics using shared feature derivation and stability
    services_summary: List[ServiceSummaryDTO] = []
    eff_threshold = (
        congestion_threshold
        if congestion_threshold is not None and congestion_threshold > 0
        else settings.CONGESTION_QUEUE_THRESHOLD
    )
    
    for s_id, s_cfg in sorted(snapshot.services.items()):
        queue_snap = snapshot.queues.get(s_id)
        q_len = len(queue_snap.waiting_tokens) if queue_snap else 0
        
        # Count counters for this service
        assigned_counters = [
            c for c in snapshot.counters.values()
            if c.assigned_service_id == s_id
        ]
        assigned_active_count = sum(1 for c in assigned_counters if c.status == "ACTIVE")
        busy_count = sum(1 for c in assigned_counters if c.status == "ACTIVE" and c.is_busy)
        unavailable_count = sum(1 for c in assigned_counters if c.status == "UNAVAILABLE")

        # Derive queue features via shared calculator
        raw_feat = compute_raw_features(snapshot, service_id=s_id, position=1)
        
        health_state = classify_stability(raw_feat.traffic_intensity)

        ttc_minutes, ttc_method = compute_time_to_congestion_fluid(
            queue_length=q_len,
            arrival_rate_per_hour=raw_feat.arrival_rate_60m,
            service_capacity_per_hour=raw_feat.service_capacity_mu,
            congestion_threshold=eff_threshold,
        )

        services_summary.append(
            ServiceSummaryDTO(
                service_id=s_id,
                service_code=s_cfg.service_code,
                service_name=s_cfg.service_name,
                queue_length=q_len,
                assigned_active_counters=assigned_active_count,
                busy_counters=busy_count,
                unavailable_counters=unavailable_count,
                arrival_rate_30m=round(raw_feat.arrival_rate_30m, 2),
                arrival_rate_60m=round(raw_feat.arrival_rate_60m, 2),
                service_mean_minutes=round(raw_feat.service_mean_minutes, 2),
                service_std_minutes=round(raw_feat.service_std_minutes, 2),
                service_capacity_mu=round(raw_feat.service_capacity_mu, 2),
                traffic_intensity=round(raw_feat.traffic_intensity, 4),
                queue_health=health_state.value,
                time_to_congestion_minutes=ttc_minutes,
                time_to_congestion_method=ttc_method,
                congestion_threshold=eff_threshold,
            )
        )

    # 3. Compile counter fleet list
    # Query counters with current token and eligibility
    db_counters = (
        db.query(Counter)
        .filter(Counter.office_id == office_id)
        .order_by(Counter.id.asc())
        .all()
    )

    counters_summary: List[CounterSummaryDTO] = []
    total_active = 0
    total_busy = 0
    total_unavailable = 0

    for c in db_counters:
        is_active = (c.status == CounterStatus.ACTIVE)
        if is_active:
            total_active += 1
        else:
            total_unavailable += 1

        # Check active token (CALLED or SERVING)
        active_tok = (
            db.query(QueueToken)
            .filter(
                QueueToken.counter_id == c.id,
                QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
            )
            .first()
        )
        is_busy = (active_tok is not None)
        if is_busy and is_active:
            total_busy += 1

        # Eligibility list
        el_ids = [cs.service_id for cs in c.counter_services]

        assigned_code = ""
        if c.assigned_service:
            assigned_code = c.assigned_service.code
        elif c.assigned_service_id in snapshot.services:
            assigned_code = snapshot.services[c.assigned_service_id].service_code

        counters_summary.append(
            CounterSummaryDTO(
                counter_id=c.id,
                name=c.name,
                assigned_service_id=c.assigned_service_id,
                assigned_service_code=assigned_code,
                status=c.status.value,
                is_busy=is_busy,
                current_token_id=active_tok.id if active_tok else None,
                current_token_number=active_tok.token_number if active_tok else None,
                eligible_service_ids=el_ids,
            )
        )

    total_waiting = sum(s.queue_length for s in services_summary)

    return AdminOperationsSummaryResponse(
        office_id=office.id,
        office_code=office.code,
        office_name=office.name,
        snapshot_time_utc=now_utc,
        services=services_summary,
        counters=counters_summary,
        total_waiting_tokens=total_waiting,
        total_active_counters=total_active,
        total_busy_counters=total_busy,
        total_unavailable_counters=total_unavailable,
    )
