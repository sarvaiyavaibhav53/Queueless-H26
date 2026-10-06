from pydantic import BaseModel, Field

class WaitTimePredictionRequest(BaseModel):
    office_id: str = Field(..., description="Office ID, e.g., RJT, AMD, STV")
    service_id: str = Field(..., description="Service ID, e.g., LL, DL_RENEWAL, VEH_REG")
    hour: int = Field(..., ge=9, le=17, description="Hour of the day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    people_ahead: int = Field(..., ge=0, description="Number of people ahead in the queue")
    queue_length: int = Field(..., ge=0, description="Total queue length")
    active_counters: int = Field(..., ge=0, description="Number of active counters")
    arrival_rate_15m: int = Field(..., ge=0, description="Arrivals in the last 15 minutes")
    counter_load: float = Field(..., ge=0, description="Queue length per active counter")
    historical_avg_service_minutes: float = Field(..., gt=0, description="Historical average service time")
    service_time_std: float = Field(..., ge=0, description="Standard deviation of service time")
    no_show_rate: float = Field(..., ge=0, le=1, description="Probability of no-shows")

class WaitTimePredictionResponse(BaseModel):
    predicted_wait_minutes: float
    model_version: str = "1.0"
    model_type: str = "RandomForestRegressor"
