from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictionSource(str, Enum):
    RF = "RF"
    ANALYTICAL_FALLBACK = "ANALYTICAL_FALLBACK"


class FallbackReason(str, Enum):
    ZERO_ACTIVE_COUNTERS = "ZERO_ACTIVE_COUNTERS"
    QUEUE_WITH_ZERO_BUSY_COUNTERS = "QUEUE_WITH_ZERO_BUSY_COUNTERS"
    PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY = "PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY"
    DEEP_QUEUE = "DEEP_QUEUE"
    RF_NON_FINITE_OUTPUT = "RF_NON_FINITE_OUTPUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class IdentityErrorCode(str, Enum):
    UNKNOWN_APPLICATION_OFFICE = "UNKNOWN_APPLICATION_OFFICE"
    UNKNOWN_APPLICATION_SERVICE = "UNKNOWN_APPLICATION_SERVICE"
    UNSUPPORTED_MODEL_OFFICE = "UNSUPPORTED_MODEL_OFFICE"
    UNSUPPORTED_MODEL_SERVICE = "UNSUPPORTED_MODEL_SERVICE"



class WaitTimePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    office_id: str = Field(..., description="Office ID, e.g., RJT, AMD, STV, MUNI, RTO")
    service_id: str = Field(..., description="Service ID, e.g., LL, DL_RENEWAL, VEH_REG, PT, WC, BC, OT, DR")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    people_ahead: int = Field(..., ge=0, description="Number of people ahead in the queue")
    queue_length: int = Field(..., ge=0, description="Total queue length")
    # Legacy fields
    active_counters: Optional[int] = Field(None, ge=0, description="Number of active counters")
    arrival_rate_15m: Optional[int] = Field(None, ge=0, description="Arrivals in the last 15 minutes")
    counter_load: Optional[float] = Field(None, ge=0, description="Queue length per active counter")
    historical_avg_service_minutes: Optional[float] = Field(None, gt=0, description="Historical average service time")
    service_time_std: Optional[float] = Field(None, ge=0, description="Standard deviation of service time")
    no_show_rate: Optional[float] = Field(None, ge=0, le=1, description="Probability of no-shows")
    # Production 15-feature contract fields (optional on legacy endpoint)
    assigned_active_counters: Optional[int] = Field(None, ge=0)
    busy_assigned_counters: Optional[int] = Field(None, ge=0)
    max_elapsed_service_minutes: Optional[float] = Field(None, ge=0)
    service_mean_minutes: Optional[float] = Field(None, gt=0)
    service_std_minutes: Optional[float] = Field(None, ge=0)
    arrival_rate_30m: Optional[float] = Field(None, ge=0)
    arrival_rate_60m: Optional[float] = Field(None, ge=0)
    service_capacity_mu: Optional[float] = Field(None, ge=0)
    traffic_intensity: Optional[float] = Field(None, ge=0)


class CanonicalPredictionResponse(BaseModel):
    """
    Canonical wait-time prediction response contract for PRAVAAH.
    Exposes empirical P10-P90 prediction interval quantiles for RF predictions,
    and null quantiles with explicit fallback_reason for analytical fallbacks.
    """
    model_config = ConfigDict(use_enum_values=True)

    predicted_wait_minutes: float = Field(..., description="Point estimate (P50 median for RF, deterministic estimate for analytical)")
    p10_minutes: Optional[float] = Field(None, description="Lower empirical prediction interval bound (P10), null on analytical fallback")
    p90_minutes: Optional[float] = Field(None, description="Upper empirical prediction interval bound (P90), null on analytical fallback")
    prediction_source: PredictionSource = Field(..., description="Closed enum: RF or ANALYTICAL_FALLBACK")
    fallback_reason: Optional[FallbackReason] = Field(None, description="Reason if analytical fallback was triggered")


class WaitTimePredictionResponse(BaseModel):
    """
    Backward-compatible response schema incorporating canonical hybrid serving fields.
    """
    model_config = ConfigDict(use_enum_values=True)

    predicted_wait_minutes: float
    p10_minutes: Optional[float] = None
    p90_minutes: Optional[float] = None
    prediction_source: PredictionSource = PredictionSource.RF
    fallback_reason: Optional[FallbackReason] = None
    model_version: str = "2.0"
    model_type: str = "HybridRFAnalytical"
