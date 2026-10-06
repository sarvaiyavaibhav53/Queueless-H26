from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.queue import TokenResponse
from app.services.queue_service import queue_service, QueueServiceError
from app.models.token import QueueToken
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tokens", tags=["tokens"])

from app.ws_manager import ws_manager
from app.api.deps import require_citizen, require_employee
from app.models.user import User

class PositionResponse(BaseModel):
    position: int

@router.get("/{token_id}", response_model=TokenResponse)
def get_token(token_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    if token.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    position = queue_service.get_queue_position(db, token.id)
    return TokenResponse(
        **token.__dict__,
        queue_position=position
    )

@router.get("/{token_id}/position", response_model=PositionResponse)
def get_token_position(token_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    if token.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    position = queue_service.get_queue_position(db, token.id)
    return PositionResponse(position=position)

@router.post("/{token_id}/cancel", response_model=TokenResponse)
def cancel_token(token_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_citizen)):
    try:
        # Check ownership first
        token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
        if not token:
            raise HTTPException(status_code=404, detail="Token not found")
        if token.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden")
            
        token = queue_service.cancel_token(db, token_id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "CANCELLED", token.office_id, token.service_id, token.id)
        return TokenResponse(**token.__dict__, queue_position=0)
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

# ----------------- STAFF APIs ----------------- #

@router.post("/{token_id}/start", response_model=TokenResponse)
def start_serving_token(token_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_employee)):
    try:
        token = queue_service.start_serving(db, token_id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "STARTED", token.office_id, token.service_id, token.id)
        return TokenResponse(**token.__dict__, queue_position=0)
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{token_id}/complete", response_model=TokenResponse)
def complete_token(token_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_employee)):
    try:
        token = queue_service.complete_token(db, token_id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "COMPLETED", token.office_id, token.service_id, token.id)
        return TokenResponse(**token.__dict__, queue_position=0)
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{token_id}/skip", response_model=TokenResponse)
def skip_token(token_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = Depends(require_employee)):
    try:
        token = queue_service.skip_token(db, token_id)
        background_tasks.add_task(ws_manager.broadcast_queue_update, "SKIPPED", token.office_id, token.service_id, token.id)
        return TokenResponse(**token.__dict__, queue_position=0)
    except QueueServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
