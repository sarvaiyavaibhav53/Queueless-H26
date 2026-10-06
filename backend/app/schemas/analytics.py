from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import date

class OverallAnalyticsResponse(BaseModel):
    total_tokens: int
    waiting_tokens: int
    called_tokens: int
    serving_tokens: int
    completed_tokens: int
    cancelled_tokens: int
    skipped_tokens: int
    average_wait_minutes: Optional[float] = None
    average_service_minutes: Optional[float] = None
    completion_rate: float
    cancellation_rate: float
    skip_rate: float

class OfficeAnalyticsResponse(BaseModel):
    office_id: int
    office_code: str
    total_tokens: int
    completed_tokens: int
    cancelled_tokens: int
    skipped_tokens: int
    current_waiting_count: int
    current_serving_count: int
    average_wait_minutes: Optional[float] = None
    average_service_minutes: Optional[float] = None

class ServiceAnalyticsResponse(BaseModel):
    service_id: int
    service_code: str
    total_tokens: int
    completed_tokens: int
    cancelled_tokens: int
    skipped_tokens: int
    current_waiting_count: int
    average_wait_minutes: Optional[float] = None
    average_service_minutes: Optional[float] = None

class QueueAnalyticsResponse(BaseModel):
    waiting_count: int
    called_count: int
    serving_count: int
    active_counters: int

class MLMonitoringResponse(BaseModel):
    total_with_predictions: int
    total_without_predictions: int
    average_predicted_wait_minutes: Optional[float] = None
    mae_minutes: Optional[float] = None
    rmse_minutes: Optional[float] = None
