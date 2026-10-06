from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.queue import JoinQueueRequest, TokenResponse, QueueStatusResponse
from app.services.queue_service import queue_service, QueueServiceError
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
import datetime

router = APIRouter(prefix="/api/v1/queues", tags=["queues"])

from app.ws_manager import ws_manager
from app.api.deps import require_citizen, require_employee
from app.models.user import User

@router.get("/{office_id}/{service_id}/status", response_model=QueueStatusResponse)
def get_queue_status(office_id: int, service_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    """
    Citizen API to get current queue status before joining.
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
    
    return QueueStatusResponse(
        waiting_count=waiting_count,
        active_counters=office.counters
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
    """
    try:
        token = queue_service.generate_token(db, request.office_id, service_id)
        token.user_id = current_user.id
        
        # Calculate position
        position = queue_service.get_queue_position(db, token.id)
        
        # ML prediction integration boundary (graceful degradation)
        # We try to call prediction_service if it works, otherwise return None or default
        predicted_wait = None
        try:
            from app.services.prediction_service import prediction_service
            import math
            office = db.query(Office).filter(Office.id == request.office_id).first()
            service = db.query(Service).filter(Service.id == service_id).first()
            if office and service:
                pred_request = {
                    "office_id": office.code,
                    "service_id": service.code,
                    "hour": datetime.datetime.now().hour,
                    "day_of_week": datetime.datetime.now().weekday(),
                    "people_ahead": position - 1,
                    "queue_length": position,
                    "active_counters": office.counters,
                    "arrival_rate_15m": 5, # Mock value as actual historical rate is complex
                    "counter_load": position / max(1, office.counters),
                    "historical_avg_service_minutes": service.mean_minutes,
                    "service_time_std": service.std_minutes or 0.0,
                    "no_show_rate": 0.05
                }
                raw_pred = prediction_service.predict_wait_time(pred_request)
                if raw_pred is not None and math.isfinite(raw_pred) and raw_pred >= 0:
                    predicted_wait = int(round(raw_pred))
        except Exception as e:
            # Graceful degradation if ML is unavailable
            # Log the exception in a real app, here we just pass
            pass
            
        token.predicted_wait_minutes = predicted_wait
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
def call_next_token(office_id: int, service_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_employee)):
    """
    Staff API to call the next waiting token.
    """
    
    try:
        token = queue_service.call_next_token(db, office_id, service_id)
        if not token:
            raise HTTPException(status_code=404, detail="No waiting tokens found in this queue")
            
        position = queue_service.get_queue_position(db, token.id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "CALLED", office_id, service_id, token.id)
        return TokenResponse(
            **token.__dict__,
            queue_position=position
        )
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
