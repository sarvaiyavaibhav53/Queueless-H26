import datetime
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.queue import (
    JoinQueueRequest,
    CallNextRequest,
    TokenResponse,
    QueueStatusResponse,
    TokenEtaResponse,
)
from app.services.queue_service import (
    queue_service,
    QueueServiceError,
    CounterNotFoundError,
    CounterConflictError,
    CounterValidationError,
)
from app.services.queue_snapshot_service import (
    get_queue_snapshot_service,
    QueueSnapshotService,
    TokenNotFoundError,
    TokenNotEligibleError,
    PointInTimeViolationError,
)
from app.services.prediction_service import IdentityValidationError
from app.models.office import Office
from app.models.service import Service
from app.models.counter import Counter, CounterStatus
from app.models.token import QueueToken, TokenStatus
from app.models.user import User, UserRole
from app.ws_manager import ws_manager
from app.api.deps import require_citizen, require_employee

logger = logging.getLogger("pravaah.api.queues")
router = APIRouter(prefix="/api/v1/queues", tags=["queues"])

@router.get("/{office_id}/{service_id}/status", response_model=QueueStatusResponse)
def get_queue_status(office_id: int, service_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    """
    Citizen API to get current queue status before joining.
    Reports waiting count and active counters assigned to the requested service.
    """
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(status_code=404, detail="Office not found")
        
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
        
    waiting_count = db.query(QueueToken).filter(
        QueueToken.office_id == office_id,
        QueueToken.service_id == service_id,
        QueueToken.status == TokenStatus.WAITING
    ).count()

    has_physical_counters = db.query(Counter).filter(Counter.office_id == office_id).first() is not None
    if has_physical_counters:
        active_counters = db.query(Counter).filter(
            Counter.office_id == office_id,
            Counter.assigned_service_id == service_id,
            Counter.status == CounterStatus.ACTIVE,
        ).count()
    else:
        # Fallback for offices without physical counter entity records (legacy test compatibility)
        active_counters = office.counters
    
    return QueueStatusResponse(
        waiting_count=waiting_count,
        active_counters=active_counters
    )

@router.get("/{office_id}/{service_id}/tokens", response_model=list[TokenResponse])
def get_queue_tokens(office_id: int, service_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_employee)):
    """
    Employee API to view current active tokens (WAITING, CALLED, SERVING).
    """
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(status_code=404, detail="Office not found")
        
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
        
    active_tokens = db.query(QueueToken).filter(
        QueueToken.office_id == office_id,
        QueueToken.service_id == service_id,
        QueueToken.status.in_([TokenStatus.WAITING, TokenStatus.CALLED, TokenStatus.SERVING])
    ).order_by(QueueToken.created_at.asc()).all()
    
    response_tokens = []
    for token in active_tokens:
        position = queue_service.get_queue_position(db, token.id) if token.status == TokenStatus.WAITING else 0
        response_tokens.append(TokenResponse(
            **token.__dict__,
            queue_position=position
        ))
        
    return response_tokens

@router.post("/{service_id}/join", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def join_queue(service_id: int, request: JoinQueueRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    """
    Citizen API to join a queue for a specific service at a specific office.
    Authoritative live ETA is retrieved via GET /api/v1/queues/{token_id}/eta.
    """
    try:
        token = queue_service.generate_token(db, request.office_id, service_id)
        token.user_id = current_user.id
        # Calculate position
        position = queue_service.get_queue_position(db, token.id)
        # Authoritative live ETA is retrieved via GET /api/v1/queues/{token_id}/eta
        # adhering to canonical 15-feature contract. Ad-hoc mock prediction is omitted.
        token.predicted_wait_minutes = None
        db.commit()
        db.refresh(token)
        
        background_tasks.add_task(ws_manager.broadcast_queue_update, "JOINED", request.office_id, service_id, token.id)
        return TokenResponse(
            **token.__dict__,
            queue_position=position
        )
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{office_id}/{service_id}/call-next", response_model=TokenResponse)
def call_next_token(
    office_id: int,
    service_id: int,
    request: Optional[CallNextRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_employee),
):
    """
    Staff API to call the next waiting token.
    Supports optional counter_id selection with server-side validation of counter availability and assignment.
    """
    counter_id = request.counter_id if request else None
    try:
        token = queue_service.call_next_token(db, office_id, service_id, counter_id=counter_id)
        if not token:
            raise HTTPException(status_code=404, detail="No waiting tokens found in this queue")
            
        position = queue_service.get_queue_position(db, token.id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "CALLED", office_id, service_id, token.id)
        return TokenResponse(
            **token.__dict__,
            queue_position=position
        )
    except CounterNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CounterConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except CounterValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{token_id}/eta", response_model=TokenEtaResponse)
def get_token_eta(
    token_id: int,
    office_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_citizen),
    snapshot_service: QueueSnapshotService = Depends(get_queue_snapshot_service),
):
    """
    Authoritative live ETA API (Step 2E.4).
    Derives all prediction state from DB snapshot -> 15 canonical features -> hybrid predictor.
    Enforces object-level authorization and office authority.
    """
    token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail=f"Token with id {token_id} not found")

    # Section 8: Token / Office Authority — token's stored office wins over path/query
    if office_id is not None and office_id != token.office_id:
        raise HTTPException(
            status_code=400,
            detail=f"Office mismatch: Token {token_id} belongs to office {token.office_id}, not {office_id}",
        )

    # Section 20: Object-level authorization
    if current_user.role == UserRole.citizen:
        if token.user_id is not None and token.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You are not authorized to view another citizen's token ETA",
            )

    try:
        return snapshot_service.get_token_eta(db=db, token_id=token_id)
    except TokenNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TokenNotEligibleError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PointInTimeViolationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IdentityValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": e.code.value,
                "message": e.message,
                "identity_value": e.identity_value,
            },
        )
    except Exception as e:
        logger.error(f"Error calculating ETA for token {token_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during live ETA calculation")
