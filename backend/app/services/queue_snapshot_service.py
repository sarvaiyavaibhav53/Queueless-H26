"""
PRAVAAH — Step 2E.4: Authoritative Queue Snapshot & Live ETA Service.

Connects the live database state to the frozen prediction system:
DB -> SimulationSnapshot -> canonical 15 features -> hybrid predictor -> ETA response.

Key Guarantees & Constraints:
- Pure server-owned state: zero client-provided prediction fields.
- Injectable clock: deterministic point-in-time calculation via explicit `now` parameter.
- Training parity: people_ahead strictly matches generate_production_dataset.py and shared/features.py.
- Point-in-time guarantee: no future records relative to injected `now` enter the snapshot.
- Single source of truth: delegates feature computation to shared/features.py and prediction
  to app/services/prediction_service.py.
- Cold-start diagnostic: detects live arrival/traffic intensity below frozen training range.
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.schemas.prediction import PredictionSource, FallbackReason
from app.schemas.queue import TokenEtaResponse
from app.services.prediction_service import (
    WaitTimePredictionService,
    get_prediction_service,
    validate_identity,
    IdentityValidationError,
)
from app.services.snapshot_adapter import (
    build_simulation_snapshot,
    SnapshotAdapterError,
    _ensure_utc,
)
from shared.features import (
    RAW_FEATURE_NAMES,
    RawFeatureVector,
    SimulationSnapshot,
    compute_raw_features,
)

logger = logging.getLogger("pravaah.queue_snapshot_service")

# Cold-start operating bounds derived from frozen production training distribution:
# - arrival_rate_60m minimum observed in training dataset: 2.0 (1st percentile: 4.0)
# - traffic_intensity minimum observed in training dataset: 0.175 (1st percentile: 0.35)
COLD_START_MIN_ARRIVAL_RATE_60M = 2.0
COLD_START_MIN_TRAFFIC_INTENSITY = 0.175


class QueueSnapshotServiceError(Exception):
    """Base error for queue snapshot service operations."""
    pass


class TokenNotFoundError(QueueSnapshotServiceError):
    """Raised when token_id does not exist in the database."""
    pass


class TokenNotEligibleError(QueueSnapshotServiceError):
    """Raised when token status or timing is not eligible for queue ETA calculation."""
    pass


class PointInTimeViolationError(QueueSnapshotServiceError):
    """Raised when a token or query violates point-in-time consistency."""
    pass


class QueueSnapshotService:
    """
    Authoritative service that extracts a point-in-time snapshot for a given token,
    constructs the canonical 15-feature vector, evaluates hybrid wait-time prediction,
    and returns a structured ETA response.
    """

    def __init__(self, prediction_service: Optional[WaitTimePredictionService] = None):
        self.prediction_service = prediction_service or get_prediction_service()

    def build_snapshot_and_features(
        self,
        db: Session,
        token_id: int,
        now: Optional[datetime] = None,
    ) -> Tuple[QueueToken, SimulationSnapshot, RawFeatureVector, int]:
        """
        Builds the domain SimulationSnapshot and 15 canonical raw features for a token at time `now`.

        Args:
            db: SQLAlchemy Session.
            token_id: Primary key of QueueToken.
            now: Explicit point-in-time timestamp (defaults to current UTC time).

        Returns:
            (token, snapshot, raw_features, position)
        """
        # 1. Normalize clock to timezone-aware UTC
        if now is None:
            now = datetime.now(timezone.utc)
        else:
            now = _ensure_utc(now, "now")

        # 2. Fetch Token
        token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
        if not token:
            raise TokenNotFoundError(f"QueueToken with id {token_id} not found")

        # 3. Point-in-time validation
        token_created_utc = _ensure_utc(token.created_at, "token.created_at")
        if token_created_utc > now:
            raise PointInTimeViolationError(
                f"Token {token_id} was created at {token_created_utc}, which is in the future relative to snapshot time {now}"
            )

        # 4. Token status eligibility check
        if token.status in (TokenStatus.COMPLETED, TokenStatus.CANCELLED, TokenStatus.SKIPPED):
            raise TokenNotEligibleError(
                f"Token {token_id} has inactive status '{token.status.value}' and is not eligible for queue ETA"
            )

        # 5. Build authoritative SimulationSnapshot via adapter
        snapshot = build_simulation_snapshot(db, office_id=token.office_id, snapshot_time_utc=now)

        # 6. Locate token in waiting queue to determine exact FIFO position
        s_queue = snapshot.queues.get(token.service_id)
        if not s_queue:
            raise QueueSnapshotServiceError(f"Service {token.service_id} queue not found in snapshot")

        # Strict parity check with training/generation pipeline:
        # For a token in WAITING status, its position is its 1-based index in the FIFO waiting list.
        # people_ahead = position - 1.
        position: Optional[int] = None
        for idx, wt in enumerate(s_queue.waiting_tokens):
            if wt.token_id == token.id:
                position = idx + 1
                break

        if position is None:
            # Token was not in waiting queue (e.g. called/serving or status transition)
            if token.status in (TokenStatus.CALLED, TokenStatus.SERVING):
                # Citizen already at desk; position is 1, people_ahead is 0
                position = 1
            else:
                raise TokenNotEligibleError(
                    f"Token {token_id} with status '{token.status.value}' not found in active waiting queue at {now}"
                )

        # 7. Compute canonical 15 features using frozen contract
        raw_features = compute_raw_features(snapshot, service_id=token.service_id, position=position)

        # 8. Strict two-tier semantic identity validation upfront
        validate_identity(raw_features.office_id, raw_features.service_id)

        return token, snapshot, raw_features, position

    def get_token_eta(
        self,
        db: Session,
        token_id: int,
        now: Optional[datetime] = None,
    ) -> TokenEtaResponse:
        """
        Calculates live wait-time ETA for a token at time `now`.

        Flow:
        DB -> build_simulation_snapshot -> compute_raw_features -> predict -> TokenEtaResponse
        """
        # 1. Normalize clock to timezone-aware UTC
        if now is None:
            now = datetime.now(timezone.utc)
        else:
            now = _ensure_utc(now, "now")

        # 2. Derive snapshot and canonical features
        token, snapshot, raw_features, position = self.build_snapshot_and_features(db, token_id, now=now)

        service_cfg = snapshot.services[token.service_id]
        office_code = snapshot.office_code
        service_code = service_cfg.service_code

        # 3. Special handling for already CALLED or SERVING tokens
        if token.status in (TokenStatus.CALLED, TokenStatus.SERVING):
            return TokenEtaResponse(
                token_id=token.id,
                token_number=token.token_number,
                office_id=token.office_id,
                office_code=office_code,
                service_id=token.service_id,
                service_code=service_code,
                predicted_wait_minutes=0.0,
                p10_minutes=0.0,
                p90_minutes=0.0,
                prediction_source=PredictionSource.ANALYTICAL_FALLBACK.value,
                fallback_reason=None,
                people_ahead=0,
                queue_length=len(snapshot.queues[token.service_id].waiting_tokens),
                is_cold_start=False,
                diagnostics=[],
                snapshot_time_utc=now,
            )

        # 4. Cold-start diagnostic detection (Section 14)
        # Evaluates if arrival rate or traffic intensity fall below trained operating range.
        # This is a diagnostic metadata field; it does NOT alter the hybrid serving policy.
        diagnostics: List[str] = []
        if raw_features.arrival_rate_60m < COLD_START_MIN_ARRIVAL_RATE_60M:
            diagnostics.append(
                f"ARRIVAL_RATE_60M_BELOW_TRAINED_MINIMUM: value {raw_features.arrival_rate_60m:.2f} < {COLD_START_MIN_ARRIVAL_RATE_60M}"
            )
        if raw_features.traffic_intensity < COLD_START_MIN_TRAFFIC_INTENSITY:
            diagnostics.append(
                f"TRAFFIC_INTENSITY_BELOW_TRAINED_MINIMUM: value {raw_features.traffic_intensity:.3f} < {COLD_START_MIN_TRAFFIC_INTENSITY}"
            )
        is_cold_start = len(diagnostics) > 0

        # 5. Invoke frozen hybrid prediction service
        pred_result = self.prediction_service.predict(raw_features)

        # 6. Return structured response contract
        return TokenEtaResponse(
            token_id=token.id,
            token_number=token.token_number,
            office_id=token.office_id,
            office_code=office_code,
            service_id=token.service_id,
            service_code=service_code,
            predicted_wait_minutes=pred_result.predicted_wait_minutes,
            p10_minutes=pred_result.p10_minutes,
            p90_minutes=pred_result.p90_minutes,
            prediction_source=pred_result.prediction_source,
            fallback_reason=pred_result.fallback_reason,
            people_ahead=raw_features.people_ahead,
            queue_length=raw_features.queue_length,
            is_cold_start=is_cold_start,
            diagnostics=diagnostics,
            snapshot_time_utc=now,
        )


queue_snapshot_service = QueueSnapshotService()


def get_queue_snapshot_service() -> QueueSnapshotService:
    return queue_snapshot_service
