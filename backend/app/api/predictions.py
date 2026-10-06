from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import WaitTimePredictionRequest, WaitTimePredictionResponse
from app.services.prediction_service import prediction_service, PredictionServiceError

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])

@router.post("/wait-time", response_model=WaitTimePredictionResponse)
def predict_wait_time(request: WaitTimePredictionRequest):
    valid_offices = {"RJT", "AMD", "STV"}
    if request.office_id not in valid_offices:
        raise HTTPException(status_code=422, detail="Invalid office_id")
        
    valid_services = {"LL", "DL_RENEWAL", "VEH_REG"}
    if request.service_id not in valid_services:
        raise HTTPException(status_code=422, detail="Invalid service_id")

    try:
        pred_minutes = prediction_service.predict_wait_time(request.dict())
        return WaitTimePredictionResponse(
            predicted_wait_minutes=pred_minutes
        )
    except PredictionServiceError as e:
        # Avoid leaking internal paths or tracebacks
        raise HTTPException(status_code=500, detail="Prediction service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
