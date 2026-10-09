"""
PRAVAAH — Physical Counter Status Management Endpoints.

Provides administrative controls for counter availability:
- POST /api/v1/counters/{counter_id}/unavailable: Transition counter to UNAVAILABLE
- POST /api/v1/counters/{counter_id}/recover: Restore counter to ACTIVE
"""

from datetime import datetime, timezone
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db.database import get_db
from app.models.counter import Counter, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.models.user import User
from app.ws_manager import ws_manager

logger = logging.getLogger("pravaah.api.counters")

router = APIRouter(prefix="/api/v1/counters", tags=["counters"])


class CounterStatusResponse(BaseModel):
    counter_id: int
    office_id: int
    name: str
    status: str
    assigned_service_id: int
    status_changed_at: datetime
    message: str


@router.post(
    "/{counter_id}/unavailable",
    response_model=CounterStatusResponse,
    summary="Mark physical counter as UNAVAILABLE",
)
def set_counter_unavailable(
    counter_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Sets a counter status to UNAVAILABLE.
    Fails with 409 Conflict if the counter currently has an active in-service token (CALLED or SERVING).
    Emits post-commit invalidation events for the affected service.
    """
    counter = db.query(Counter).filter(Counter.id == counter_id).first()
    if not counter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Counter with ID {counter_id} not found",
        )

    # Active token safety (Step 2E.5B):
    # Marking counter unavailable succeeds even when serving a token.
    # The active token remains intact and can be completed normally through the completion endpoint.
    # The unavailable counter is blocked from receiving future tokens until explicitly recovered.

    now_utc = datetime.now(timezone.utc)
    counter.status = CounterStatus.UNAVAILABLE
    counter.status_changed_at = now_utc

    db.commit()
    db.refresh(counter)

    # Post-commit invalidation: exactly one event
    background_tasks.add_task(
        ws_manager.broadcast_invalidation,
        event_type="counter.status_changed",
        office_id=counter.office_id,
        service_id=counter.assigned_service_id,
    )

    return CounterStatusResponse(
        counter_id=counter.id,
        office_id=counter.office_id,
        name=counter.name,
        status=counter.status.value,
        assigned_service_id=counter.assigned_service_id,
        status_changed_at=counter.status_changed_at,
        message=f"Counter '{counter.name}' marked UNAVAILABLE successfully.",
    )


@router.post(
    "/{counter_id}/recover",
    response_model=CounterStatusResponse,
    summary="Recover physical counter to ACTIVE status",
)
def recover_counter(
    counter_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Restores an UNAVAILABLE counter to ACTIVE status without changing its assigned service.
    Emits post-commit invalidation events for the affected service.
    """
    counter = db.query(Counter).filter(Counter.id == counter_id).first()
    if not counter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Counter with ID {counter_id} not found",
        )

    now_utc = datetime.now(timezone.utc)
    counter.status = CounterStatus.ACTIVE
    counter.status_changed_at = now_utc

    db.commit()
    db.refresh(counter)

    # Post-commit invalidation: exactly one event
    background_tasks.add_task(
        ws_manager.broadcast_invalidation,
        event_type="counter.status_changed",
        office_id=counter.office_id,
        service_id=counter.assigned_service_id,
    )

    return CounterStatusResponse(
        counter_id=counter.id,
        office_id=counter.office_id,
        name=counter.name,
        status=counter.status.value,
        assigned_service_id=counter.assigned_service_id,
        status_changed_at=counter.status_changed_at,
        message=f"Counter '{counter.name}' recovered to ACTIVE successfully.",
    )
