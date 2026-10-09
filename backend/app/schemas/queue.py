from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
from app.models.token import TokenStatus

class JoinQueueRequest(BaseModel):
    office_id: int

class CallNextRequest(BaseModel):
    counter_id: Optional[int] = None

class QueueStatusResponse(BaseModel):
    waiting_count: int
    active_counters: int

class TokenResponse(BaseModel):
    id: int
    token_number: str
    service_id: int
    office_id: int
    status: TokenStatus
    counter_id: Optional[int] = None
    queue_position: Optional[int] = None
    predicted_wait_minutes: Optional[float] = None
    created_at: datetime
    called_at: Optional[datetime] = None
    serving_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    skipped_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenEtaResponse(BaseModel):
    """
    Authoritative live ETA response contract for PRAVAAH (Step 2E.4).
    Reuses existing PredictionSource and FallbackReason enums.
    Empirical P10-P90 prediction intervals from frozen quantile regressors.
    """
    model_config = ConfigDict(use_enum_values=True)

    token_id: int
    token_number: str
    office_id: int
    office_code: str
    service_id: int
    service_code: str
    predicted_wait_minutes: float
    p10_minutes: Optional[float] = None
    p90_minutes: Optional[float] = None
    prediction_source: str
    fallback_reason: Optional[str] = None
    people_ahead: int
    queue_length: int
    is_cold_start: bool = False
    diagnostics: list[str] = []
    snapshot_time_utc: datetime
    provenance: str = "Empirical P10–P90 prediction intervals from quantile models, not statistical confidence intervals."
