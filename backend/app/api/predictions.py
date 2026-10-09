from fastapi import APIRouter, HTTPException, Depends
from app.api.deps import require_citizen
from app.models.user import User
from app.schemas.prediction import WaitTimePredictionRequest, WaitTimePredictionResponse
from app.services.prediction_service import (
    WaitTimePredictionService,
    get_prediction_service,
    PredictionServiceError,
    IdentityValidationError,
    validate_identity,
)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])

@router.post("/wait-time", response_model=WaitTimePredictionResponse)
def predict_wait_time(
    request: WaitTimePredictionRequest,
    current_user: User = Depends(require_citizen),
    service: WaitTimePredictionService = Depends(get_prediction_service),
):
    # Strict two-tier semantic identity validation:
    # Tier 1: unknown application identity -> 422 UNKNOWN_APPLICATION_*
    # Tier 2: known application identity but unsupported by frozen model -> 422 UNSUPPORTED_MODEL_*
    try:
        validate_identity(request.office_id, request.service_id)
    except IdentityValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": e.code.value,
                "message": e.message,
                "identity_value": e.identity_value,
            },
        )

    try:
        req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        canonical_res = service.predict(req_dict)
        return WaitTimePredictionResponse(
            predicted_wait_minutes=canonical_res.predicted_wait_minutes,
            p10_minutes=canonical_res.p10_minutes,
            p90_minutes=canonical_res.p90_minutes,
            prediction_source=canonical_res.prediction_source,
            fallback_reason=canonical_res.fallback_reason,
        )
    except IdentityValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": e.code.value,
                "message": e.message,
                "identity_value": e.identity_value,
            },
        )
    except PredictionServiceError as e:
        raise HTTPException(status_code=500, detail="Prediction service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during prediction")



