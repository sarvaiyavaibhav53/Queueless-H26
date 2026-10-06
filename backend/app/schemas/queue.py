from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from app.models.token import TokenStatus

class JoinQueueRequest(BaseModel):
    office_id: int

class QueueStatusResponse(BaseModel):
    waiting_count: int
    active_counters: int

class TokenResponse(BaseModel):
    id: int
    token_number: str
    service_id: int
    office_id: int
    status: TokenStatus
    queue_position: Optional[int] = None
    predicted_wait_minutes: Optional[float] = None
    created_at: datetime
    called_at: Optional[datetime] = None
    serving_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    skipped_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
