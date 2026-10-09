"""
PRAVAAH — Step 2E.3: Frozen ML Hybrid Serving Service.

Implements safe, deterministic hybrid serving for wait-time predictions:
- Consumes frozen Random Forest model (ml/models/production_wait_time_rf.joblib, SHA-256 verified).
- Consumes frozen Quantile Regressors (P10, P50, P90) for empirical prediction intervals.
- Enforces exact canonical 15-feature contract in RAW_FEATURE_NAMES sequence.
- Implements hybrid fallback policy (Guards A, B, C, D) before RF invocation.
- Deterministic residual-aware analytical fallback when unsupported states are encountered.
- Validates model output sanity (negative clipping, non-finite handling, quantile monotonicity).
- Thread-safe singleton model loading lifecycle with checksum verification.
"""

import hashlib
import logging
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from shared.features import (
    RAW_FEATURE_NAMES,
    RawFeatureVector,
    SimulationSnapshot,
    compute_raw_features,
)
from app.schemas.prediction import PredictionSource, FallbackReason, IdentityErrorCode

logger = logging.getLogger("pravaah.prediction_service")

# =============================================================================
# 1. FROZEN ARTIFACT & SERVING BOUNDARY CONSTANTS
# =============================================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Frozen Production Model Paths
FROZEN_RF_MODEL_PATH = os.path.join(REPO_ROOT, "ml", "models", "production_wait_time_rf.joblib")
FROZEN_RF_SHA256 = "99de21b0170fb856f4a8fdeaf70c8d1427c1e6739c9ad2a7e72c00c8a6a28af7"

FROZEN_Q10_MODEL_PATH = os.path.join(REPO_ROOT, "ml", "models", "production_quantile_p10.joblib")
FROZEN_Q50_MODEL_PATH = os.path.join(REPO_ROOT, "ml", "models", "production_quantile_p50.joblib")
FROZEN_Q90_MODEL_PATH = os.path.join(REPO_ROOT, "ml", "models", "production_quantile_p90.joblib")

# Legacy Model Path (retained for isolated legacy test support only)
LEGACY_RF_MODEL_PATH = os.path.join(REPO_ROOT, "ml", "models", "random_forest_wait_time_pipeline.joblib")

# Application Identity Taxonomy (recognized by backend database/APIs)
APPLICATION_OFFICES = frozenset({"MUNI", "RTO", "RJT", "AMD", "STV"})
APPLICATION_SERVICES = frozenset({"PT", "WC", "BC", "OT", "DR", "LL", "DL_RENEWAL", "VEH_REG"})

# Model-Supported Identity Space (derived directly from frozen production Random Forest v2.0 artifact)
MODEL_SUPPORTED_OFFICES = frozenset({"MUNI", "RTO"})
MODEL_SUPPORTED_SERVICES = frozenset({"BC", "DR", "OT", "PT", "WC"})

# Model-Supported Service Baseline Parameters (mean and std in minutes from frozen scenarios)
MODEL_SUPPORTED_SERVICE_PARAMS: Dict[str, Dict[str, float]] = {
    "PT": {"mean": 10.0, "std": 4.0},
    "WC": {"mean": 15.0, "std": 6.0},
    "BC": {"mean": 7.0, "std": 2.8},
    "OT": {"mean": 14.0, "std": 5.6},
    "DR": {"mean": 8.0, "std": 3.2},
}

# Backward compatibility aliases
VALID_OFFICE_CODES = APPLICATION_OFFICES
VALID_SERVICE_CODES = APPLICATION_SERVICES

# Serving-safety threshold selected from the frozen Step 2E.2A serving-boundary audit:
# for people_ahead >= 11, the residual analytical method outperformed RF on both validation
# and test subsets. This is a serving-policy threshold, not a training threshold.
DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD = 11

CANONICAL_15_FEATURES: Tuple[str, ...] = RAW_FEATURE_NAMES

LEGACY_12_FEATURES = [
    "office_id", "service_id", "hour", "day_of_week", "people_ahead",
    "queue_length", "active_counters", "arrival_rate_15m", "counter_load",
    "historical_avg_service_minutes", "service_time_std", "no_show_rate"
]


# =============================================================================
# 2. RESPONSE CONTRACT & EXCEPTIONS
# =============================================================================

class PredictionServiceError(Exception):
    """Base exception for prediction service failures."""
    pass


class IdentityValidationError(PredictionServiceError):
    """
    Raised when an office or service identity is unknown or unsupported by the frozen model.
    Carries explicit IdentityErrorCode for HTTP 422 mapping.
    """
    def __init__(self, code: IdentityErrorCode, message: str, identity_value: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.identity_value = identity_value


def validate_identity(office_id: str, service_id: str) -> None:
    """
    Strict two-tier identity validation contract:
    Tier 1: Application Taxonomy (unknown office / service) -> 422 UNKNOWN_APPLICATION_*
    Tier 2: Frozen Model Contract (known application office / service not learned by model) -> 422 UNSUPPORTED_MODEL_*
    """
    # Tier 1: Application Taxonomy Check
    if office_id not in APPLICATION_OFFICES:
        raise IdentityValidationError(
            code=IdentityErrorCode.UNKNOWN_APPLICATION_OFFICE,
            message=f"Unknown application office_id '{office_id}'. Known application offices: {sorted(APPLICATION_OFFICES)}",
            identity_value=office_id,
        )

    if service_id not in APPLICATION_SERVICES:
        raise IdentityValidationError(
            code=IdentityErrorCode.UNKNOWN_APPLICATION_SERVICE,
            message=f"Unknown application service_id '{service_id}'. Known application services: {sorted(APPLICATION_SERVICES)}",
            identity_value=service_id,
        )

    # Tier 2: Frozen Model Support Contract Check
    if office_id not in MODEL_SUPPORTED_OFFICES:
        raise IdentityValidationError(
            code=IdentityErrorCode.UNSUPPORTED_MODEL_OFFICE,
            message=(
                f"Application office '{office_id}' is not supported by frozen model v2.0. "
                f"Supported model offices: {sorted(MODEL_SUPPORTED_OFFICES)}. Model retraining required."
            ),
            identity_value=office_id,
        )

    if service_id not in MODEL_SUPPORTED_SERVICES:
        raise IdentityValidationError(
            code=IdentityErrorCode.UNSUPPORTED_MODEL_SERVICE,
            message=(
                f"Application service '{service_id}' is not supported by frozen model v2.0. "
                f"Supported model services: {sorted(MODEL_SUPPORTED_SERVICES)}. Model retraining required."
            ),
            identity_value=service_id,
        )



@dataclass(frozen=True)
class CanonicalPredictionResult:
    """
    Canonical wait-time prediction response contract for PRAVAAH.
    Exposes empirical P10-P90 prediction interval quantiles for RF predictions,
    and null quantiles with explicit fallback_reason for analytical fallbacks.
    """
    predicted_wait_minutes: float
    p10_minutes: Optional[float]
    p90_minutes: Optional[float]
    prediction_source: str  # Closed enum value: "RF" or "ANALYTICAL_FALLBACK"
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_wait_minutes": self.predicted_wait_minutes,
            "p10_minutes": self.p10_minutes,
            "p90_minutes": self.p90_minutes,
            "prediction_source": self.prediction_source,
            "fallback_reason": self.fallback_reason,
        }


# =============================================================================
# 3. DETERMINISTIC RESIDUAL-AWARE ANALYTICAL METHOD
# =============================================================================

def compute_analytical_residual_aware(features: Dict[str, Any]) -> float:
    """
    Deterministic residual-aware analytical estimator derived from queuing theory:
    base_wait = (k * mean_s + residual) / max(1, c)
    where residual = max(0, mean_s - elapsed) if busy_counters >= 1 else 0.0.
    Zero fitted parameters, zero test-set leakage, transparent linear scaling.

    Capacity semantics:
    When there is a waiting queue (queue_length >= 1) and partially unavailable capacity
    (busy >= 1 and busy < assigned), operational serviceable capacity is limited to busy counters.
    Using assigned counters would overstate capacity by including desks that are
    assigned but not busy with waiting queue (assigned_not_busy_with_waiting_queue).
    """
    k = float(features.get("people_ahead") or 0.0)
    mean_s = float(features.get("service_mean_minutes") or features.get("historical_avg_service_minutes") or 10.0)
    assigned = float(features.get("assigned_active_counters") or features.get("active_counters") or 1.0)
    elapsed = max(0.0, float(features.get("max_elapsed_service_minutes") or 0.0))
    busy = float(features.get("busy_assigned_counters") or 0.0)
    queue_length = float(features.get("queue_length") or 0.0)

    # Use only currently serviceable capacity during partial availability (assigned_not_busy_with_waiting_queue)
    if queue_length >= 1.0 and busy >= 1.0 and busy < assigned:
        c = max(1.0, busy)
    else:
        c = max(1.0, assigned)

    has_busy = busy >= 1.0
    residual = max(0.0, mean_s - elapsed) if has_busy else 0.0
    pred = (k * mean_s + residual) / c
    return max(0.0, float(pred))


# =============================================================================
# 4. HYBRID SERVING BOUNDARY POLICY
# =============================================================================

def evaluate_hybrid_serving_policy(features: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Evaluates serving-safety guards strictly BEFORE invoking the Random Forest model.

    Returns:
        (use_fallback: bool, fallback_reason: Optional[str])
    """
    queue_length = int(features.get("queue_length") or 0)
    busy = int(features.get("busy_assigned_counters") or 0)
    assigned = int(features.get("assigned_active_counters") or features.get("active_counters") or 0)
    people_ahead = int(features.get("people_ahead") or 0)

    # Guard B — no active assigned counters
    if assigned == 0:
        return True, FallbackReason.ZERO_ACTIVE_COUNTERS.value

    # Guard A — waiting queue but no busy assigned counter
    elif queue_length >= 1 and busy == 0:
        return True, FallbackReason.QUEUE_WITH_ZERO_BUSY_COUNTERS.value

    # Guard C — waiting queue with partially unavailable assigned capacity
    elif queue_length >= 1 and busy < assigned:
        return True, FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value

    # Guard D — deep queue
    elif people_ahead >= DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD:
        return True, FallbackReason.DEEP_QUEUE.value

    else:
        return False, None



# =============================================================================
# 5. WAIT TIME PREDICTION SERVICE
# =============================================================================

class WaitTimePredictionService:
    """
    Authoritative backend prediction service for PRAVAAH.
    Features:
    - Thread-safe singleton lifecycle.
    - Checksum verification for the frozen production Random Forest model.
    - Pre-inference hybrid serving policy (Guards A, B, C, D).
    - Monotonic quantile intervals (P10 <= P50 <= P90).
    - Anomaly clipping & non-finite fallback.
    - Strict 15-feature contract preservation.
    """
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super(WaitTimePredictionService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._class_lock:
            if getattr(self, "_initialized", False):
                return
            self._lock = threading.Lock()
            self.rf_model = None
            self.q10_model = None
            self.q50_model = None
            self.q90_model = None
            self.legacy_model = None
            self.model = None
            self.rf_available = False
            self.is_legacy_loaded = False
            self.last_load_error: Optional[str] = None
            self.features_list = list(LEGACY_12_FEATURES)
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml_serving")
            self._initialized = True

    def verify_artifact_checksum(self, model_path: str, expected_sha256: str) -> bool:
        """Verifies SHA-256 hash of the artifact on disk."""
        if not os.path.exists(model_path):
            return False
        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest().lower()
        return digest == expected_sha256.lower()

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Loads the production Random Forest and Quantile models, verifying checksums.
        Thread-safe; called at startup/lifespan or lazily on first prediction request.
        """
        with self._lock:
            # If explicit model path provided (e.g. from legacy tests), load it
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise PredictionServiceError(f"Model file not found at {model_path}")
                try:
                    loaded = joblib.load(model_path)
                    # Check if it's the production model or legacy model
                    if os.path.abspath(model_path) == os.path.abspath(FROZEN_RF_MODEL_PATH):
                        self.rf_model = loaded
                        self.rf_available = True
                        self.is_legacy_loaded = False
                        self._load_quantiles()
                    else:
                        self.legacy_model = loaded
                        self.is_legacy_loaded = True
                    self.model = loaded  # backward compatibility alias
                    return
                except Exception as e:
                    self.last_load_error = str(e)
                    raise PredictionServiceError(f"Failed to load model from {model_path}: {e}")

            # Default: Load frozen production wait-time RF model
            if not os.path.exists(FROZEN_RF_MODEL_PATH):
                err_msg = f"Production RF model missing at {FROZEN_RF_MODEL_PATH}"
                logger.error(err_msg)
                self.rf_available = False
                self.last_load_error = err_msg
                return

            if not self.verify_artifact_checksum(FROZEN_RF_MODEL_PATH, FROZEN_RF_SHA256):
                err_msg = (
                    f"Production RF model checksum mismatch at {FROZEN_RF_MODEL_PATH}! "
                    f"Expected SHA-256: {FROZEN_RF_SHA256}."
                )
                logger.error(err_msg)
                self.rf_available = False
                self.last_load_error = err_msg
                return

            try:
                self.rf_model = joblib.load(FROZEN_RF_MODEL_PATH)
                self.model = self.rf_model  # backward compatibility
                self.rf_available = True
                self.is_legacy_loaded = False

                # Extract and verify categories directly from the loaded frozen pipeline
                cat_tx = self.rf_model.named_steps["preprocessor"].named_transformers_["cat"]
                artifact_offices = frozenset(cat_tx.categories_[0])
                artifact_services = frozenset(cat_tx.categories_[1])
                if artifact_offices != MODEL_SUPPORTED_OFFICES:
                    raise PredictionServiceError(
                        f"Frozen artifact office categories mismatch! Found: {artifact_offices}, Expected: {MODEL_SUPPORTED_OFFICES}"
                    )
                if artifact_services != MODEL_SUPPORTED_SERVICES:
                    raise PredictionServiceError(
                        f"Frozen artifact service categories mismatch! Found: {artifact_services}, Expected: {MODEL_SUPPORTED_SERVICES}"
                    )
                self.artifact_supported_offices = artifact_offices
                self.artifact_supported_services = artifact_services

                self._load_quantiles()
                logger.info("Frozen Production Random Forest loaded successfully (SHA-256 and categories verified).")
            except Exception as e:
                err_msg = f"Failed to load production RF model: {e}"
                logger.error(err_msg)
                self.rf_available = False
                self.last_load_error = err_msg

    def _load_quantiles(self) -> None:
        """Loads optional quantile regressors if present on disk."""
        try:
            if "_loss" not in sys.modules:
                try:
                    import sklearn._loss._loss
                    sys.modules["_loss"] = sklearn._loss._loss
                except Exception:
                    pass
            if os.path.exists(FROZEN_Q10_MODEL_PATH):
                self.q10_model = joblib.load(FROZEN_Q10_MODEL_PATH)
            if os.path.exists(FROZEN_Q50_MODEL_PATH):
                self.q50_model = joblib.load(FROZEN_Q50_MODEL_PATH)
            if os.path.exists(FROZEN_Q90_MODEL_PATH):
                self.q90_model = joblib.load(FROZEN_Q90_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Quantile models could not be loaded: {e}. RF will serve point predictions only.")

    def ensure_loaded(self) -> None:
        """Ensures the model is loaded lazily if not loaded during startup."""
        if not self.rf_available and not self.is_legacy_loaded and self.rf_model is None:
            self.load_model()

    def _normalize_to_15_features(
        self,
        input_data: Union[Dict[str, Any], RawFeatureVector, SimulationSnapshot],
        service_id: Optional[int] = None,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Converts diverse input types into a normalized 15-feature dictionary."""
        if isinstance(input_data, RawFeatureVector):
            validate_identity(input_data.office_id, input_data.service_id)
            return input_data.to_dict()

        if isinstance(input_data, SimulationSnapshot):
            if service_id is None or position is None:
                raise ValueError("service_id and position must be provided when snapshot is passed")
            raw_vec = compute_raw_features(input_data, service_id=service_id, position=position)
            validate_identity(raw_vec.office_id, raw_vec.service_id)
            return raw_vec.to_dict()

        if isinstance(input_data, dict):
            # 1. Validate office_id and service_id upfront (strict two-tier contract)
            office_val = str(input_data.get("office_id", "")).strip()
            service_val = str(input_data.get("service_id", "")).strip()
            validate_identity(office_val, service_val)

            # Check if input already has all 15 non-null features
            has_15 = all(f in input_data and input_data[f] is not None for f in CANONICAL_15_FEATURES)
            if has_15:
                res = {f: input_data[f] for f in CANONICAL_15_FEATURES}
                res["office_id"] = office_val
                res["service_id"] = service_val
                return res

            # Map legacy or partial features gracefully using model-supported service baseline
            norm = {k: v for k, v in input_data.items() if v is not None}
            c_active = int(norm.get("assigned_active_counters") or norm.get("active_counters") or 1)
            norm["assigned_active_counters"] = c_active
            q_len = int(norm.get("queue_length") or 0)
            norm["busy_assigned_counters"] = int(norm.get("busy_assigned_counters") if norm.get("busy_assigned_counters") is not None else (c_active if q_len > 0 else 0))
            norm["max_elapsed_service_minutes"] = float(norm.get("max_elapsed_service_minutes") if norm.get("max_elapsed_service_minutes") is not None else 0.0)
            params = MODEL_SUPPORTED_SERVICE_PARAMS[service_val]
            s_mean = float(norm.get("service_mean_minutes") or norm.get("historical_avg_service_minutes") or params["mean"])
            norm["service_mean_minutes"] = s_mean
            norm["service_std_minutes"] = float(norm.get("service_std_minutes") or norm.get("service_time_std") or params["std"])
            arr_15 = float(norm.get("arrival_rate_15m") or 5.0)
            norm["arrival_rate_30m"] = float(norm.get("arrival_rate_30m") if norm.get("arrival_rate_30m") is not None else arr_15 * 2.0)
            norm["arrival_rate_60m"] = float(norm.get("arrival_rate_60m") if norm.get("arrival_rate_60m") is not None else arr_15 * 4.0)
            mu = float(c_active) * (60.0 / s_mean) if s_mean > 0 else 0.0
            norm["service_capacity_mu"] = float(norm.get("service_capacity_mu") if norm.get("service_capacity_mu") is not None else mu)
            norm["traffic_intensity"] = float(norm.get("traffic_intensity") if norm.get("traffic_intensity") is not None else (norm["arrival_rate_60m"] / mu if mu > 0 else 0.0))
            norm["people_ahead"] = int(norm.get("people_ahead") or 0)
            norm["queue_length"] = q_len
            norm["hour"] = int(norm.get("hour") or 10)
            norm["day_of_week"] = int(norm.get("day_of_week") or 0)
            norm["office_id"] = office_val
            norm["service_id"] = service_val
            return {f: norm[f] for f in CANONICAL_15_FEATURES}

        raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

    def predict(
        self,
        input_data: Union[Dict[str, Any], RawFeatureVector, SimulationSnapshot],
        service_id: Optional[int] = None,
        position: Optional[int] = None,
    ) -> CanonicalPredictionResult:
        """
        Executes the canonical hybrid serving pipeline:
        1. Normalize features to canonical 15 features and validate office/service.
        2. Evaluate serving policy guards (Guards A, B, C, D).
        3. If guard triggers or model unavailable: execute analytical fallback.
        4. Else: invoke frozen RF pipeline and quantile models.
        5. Validate output sanity (clipping, non-finite handling, monotonic quantiles).
        """
        self.ensure_loaded()
        features = self._normalize_to_15_features(input_data, service_id=service_id, position=position)

        # 1. Check Model Availability
        if not self.rf_available:
            reason = FallbackReason.MODEL_UNAVAILABLE.value
            logger.warning(f"RF model unavailable. Invoking analytical fallback: {reason}")
            analytical_val = compute_analytical_residual_aware(features)
            return CanonicalPredictionResult(
                predicted_wait_minutes=round(analytical_val, 4),
                p10_minutes=None,
                p90_minutes=None,
                prediction_source=PredictionSource.ANALYTICAL_FALLBACK.value,
                fallback_reason=reason,
            )

        # 2. Evaluate Serving Policy Guards BEFORE invoking RF
        use_fallback, guard_reason = evaluate_hybrid_serving_policy(features)
        if use_fallback:
            logger.info(f"Hybrid serving guard triggered: {guard_reason}. Invoking analytical fallback.")
            analytical_val = compute_analytical_residual_aware(features)
            return CanonicalPredictionResult(
                predicted_wait_minutes=round(analytical_val, 4),
                p10_minutes=None,
                p90_minutes=None,
                prediction_source=PredictionSource.ANALYTICAL_FALLBACK.value,
                fallback_reason=guard_reason,
            )

        # 3. Construct exact DataFrame with ONLY the canonical 15 features in exact order
        row_values = [features[col] for col in CANONICAL_15_FEATURES]
        df_input = pd.DataFrame([row_values], columns=list(CANONICAL_15_FEATURES))

        try:
            # Direct sequential inference avoids GIL thread-synchronization overhead for single-row prediction
            rf_point = float(self.rf_model.predict(df_input)[0])
            p50 = float(self.q50_model.predict(df_input)[0]) if self.q50_model is not None else rf_point
            p10: Optional[float] = float(self.q10_model.predict(df_input)[0]) if self.q10_model is not None else None
            p90: Optional[float] = float(self.q90_model.predict(df_input)[0]) if self.q90_model is not None else None

        except Exception as e:
            logger.error(f"Inference error in RF model: {e}. Falling back to analytical method.")
            analytical_val = compute_analytical_residual_aware(features)
            return CanonicalPredictionResult(
                predicted_wait_minutes=round(analytical_val, 4),
                p10_minutes=None,
                p90_minutes=None,
                prediction_source=PredictionSource.ANALYTICAL_FALLBACK.value,
                fallback_reason=FallbackReason.MODEL_UNAVAILABLE.value,
            )

        # 5. Output Validation: Check Non-Finite Outputs (NaN / +inf / -inf)
        is_finite = math.isfinite(p50) and (p10 is None or math.isfinite(p10)) and (p90 is None or math.isfinite(p90))
        if not is_finite:
            logger.error(
                f"Non-finite output detected from RF model (p50={p50}, p10={p10}, p90={p90}). "
                "Triggering analytical fallback."
            )
            analytical_val = compute_analytical_residual_aware(features)
            return CanonicalPredictionResult(
                predicted_wait_minutes=round(analytical_val, 4),
                p10_minutes=None,
                p90_minutes=None,
                prediction_source=PredictionSource.ANALYTICAL_FALLBACK.value,
                fallback_reason=FallbackReason.RF_NON_FINITE_OUTPUT.value,
            )

        # 6. Output Validation: Clip Negative Finite Predictions
        if p50 < 0.0:
            logger.warning(f"Negative prediction {p50} clipped to 0.0")
            p50 = 0.0
        if p10 is not None and p10 < 0.0:
            logger.warning(f"Negative P10 quantile {p10} clipped to 0.0")
            p10 = 0.0
        if p90 is not None and p90 < 0.0:
            logger.warning(f"Negative P90 quantile {p90} clipped to 0.0")
            p90 = 0.0

        # 7. Output Validation: Monotonic Quantile Correction (P10 <= P50 <= P90)
        if p10 is not None and p90 is not None:
            if not (p10 <= p50 <= p90):
                logger.warning(
                    f"Quantile crossing anomaly detected: P10={p10:.2f}, P50={p50:.2f}, P90={p90:.2f}. "
                    "Applying deterministic monotonic correction."
                )
                p50 = max(p10, p50)
                p90 = max(p50, p90)

        return CanonicalPredictionResult(
            predicted_wait_minutes=round(p50, 4),
            p10_minutes=round(p10, 4) if p10 is not None else None,
            p90_minutes=round(p90, 4) if p90 is not None else None,
            prediction_source=PredictionSource.RF.value,
            fallback_reason=None,
        )

    def predict_wait_time(self, data: Dict[str, Any]) -> float:
        """
        Backward-compatible prediction method returning a single float wait-time value.
        Supports legacy 12-feature requests when legacy model is explicitly loaded in test isolation,
        while defaulting to canonical hybrid serving for production workflows.
        """
        # If legacy model is explicitly loaded (as in legacy tests), serve via legacy pipeline
        if self.is_legacy_loaded and self.legacy_model is not None:
            try:
                input_df = pd.DataFrame([[data[f] for f in self.features_list]], columns=self.features_list)
                prediction = self.legacy_model.predict(input_df)[0]
                return float(prediction)
            except Exception as e:
                raise PredictionServiceError(f"Legacy prediction failed: {e}")

        # Otherwise serve through authoritative canonical hybrid pipeline
        res = self.predict(data)
        return res.predicted_wait_minutes


# Canonical Singleton Instance
prediction_service = WaitTimePredictionService()


def get_prediction_service() -> WaitTimePredictionService:
    """Dependency provider returning the authoritative prediction service singleton."""
    return prediction_service
