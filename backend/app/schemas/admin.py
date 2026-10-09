"""
PRAVAAH — Admin Command-Center & Scenario Operations Schemas.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ScenarioLoadRequest(BaseModel):
    scenario_id: str
    now: Optional[datetime] = None


class ScenarioLoadResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    office_id: int
    office_code: str
    affected_service_ids: List[int]
    counters_loaded: int
    waiting_tokens_loaded: int
    busy_counters_loaded: int
    loaded_at_utc: datetime


class AvailableScenariosResponse(BaseModel):
    scenarios: List[dict]


class CounterSummaryDTO(BaseModel):
    counter_id: int
    name: str
    assigned_service_id: int
    assigned_service_code: str
    status: str
    is_busy: bool
    current_token_id: Optional[int] = None
    current_token_number: Optional[str] = None
    eligible_service_ids: List[int]


class ServiceSummaryDTO(BaseModel):
    service_id: int
    service_code: str
    service_name: str
    queue_length: int
    assigned_active_counters: int
    busy_counters: int
    unavailable_counters: int
    arrival_rate_30m: float
    arrival_rate_60m: float
    service_mean_minutes: float
    service_std_minutes: float
    service_capacity_mu: float
    traffic_intensity: float
    queue_health: str  # STABLE | WATCH | UNSTABLE
    time_to_congestion_minutes: Optional[float] = None
    time_to_congestion_method: Optional[str] = "fluid_approximation"
    congestion_threshold: Optional[int] = None


class AdminOperationsSummaryResponse(BaseModel):
    office_id: int
    office_code: str
    office_name: str
    snapshot_time_utc: datetime
    services: List[ServiceSummaryDTO]
    counters: List[CounterSummaryDTO]
    total_waiting_tokens: int
    total_active_counters: int
    total_busy_counters: int
    total_unavailable_counters: int
