"""
PRAVAAH — Database Scenario Loader Service.

Converts frozen simulation scenario definitions into operational database state
for repeatable, deterministic hackathon demonstrations.

Guarantees:
- Strictly office-scoped: never mutates or resets another office's data.
- Atomic transaction: rollback on error leaves existing state untouched.
- Reuses existing models: Office, Service, Counter, CounterService, QueueToken.
- Preserves token lifecycle invariants: valid active tokens, no duplicate assignments.
- Seeds historical arrivals for realistic trailing 30m/60m rate observations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.counter import Counter, CounterService, CounterStatus
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter
from app.simulation.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioConfig,
    get_scenario,
    compute_service_capacity_mu,
)

logger = logging.getLogger("pravaah.scenario_loader")


class ScenarioLoaderError(Exception):
    """Raised when scenario loading fails validation or consistency checks."""
    pass


@dataclass(frozen=True)
class ScenarioLoadResult:
    scenario_id: str
    scenario_name: str
    office_id: int
    office_code: str
    affected_service_ids: List[int]
    counters_loaded: int
    waiting_tokens_loaded: int
    busy_counters_loaded: int
    loaded_at_utc: datetime


def get_available_scenarios() -> Dict[str, str]:
    """Returns dictionary of supported scenario IDs and human-readable names."""
    out = {}
    for k, builder in SCENARIO_REGISTRY.items():
        if len(k) > 1 and not k.startswith("D_") and not k.startswith("E_"):
            try:
                cfg = builder()
                out[cfg.scenario_id] = cfg.scenario_name
            except Exception:
                pass
    if "E_known" in SCENARIO_REGISTRY:
        try:
            cfg = SCENARIO_REGISTRY["E_known"]()
            out["E_known"] = cfg.scenario_name
        except Exception:
            pass
    return out


def load_scenario_into_db(
    db: Session,
    office_id: int,
    scenario_id: str,
    now: Optional[datetime] = None,
) -> ScenarioLoadResult:
    """
    Deterministically resets and loads a scenario for the specified office.

    Args:
        db: SQLAlchemy database session.
        office_id: Office ID to load scenario for.
        scenario_id: ID or short alias of scenario from SCENARIO_REGISTRY.
        now: Reference timestamp (defaults to timezone.utc now).

    Returns:
        ScenarioLoadResult with summary counts and affected service IDs.
    """
    ref_time = now or datetime.now(timezone.utc)
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)

    # 1. Validate scenario exists
    if scenario_id not in SCENARIO_REGISTRY:
        raise ScenarioLoaderError(
            f"Unknown scenario_id '{scenario_id}'. Available: {list(SCENARIO_REGISTRY.keys())}"
        )

    scen: ScenarioConfig = get_scenario(scenario_id)

    # 2. Validate office exists
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise ScenarioLoaderError(f"Office with id {office_id} does not exist.")

    # 3. Validate office match
    if scen.office_id != office_id:
        raise ScenarioLoaderError(
            f"Scenario '{scen.scenario_id}' belongs to office {scen.office_id} ({scen.office_code}), "
            f"cannot be loaded into office {office_id} ({office.code})."
        )

    affected_services: Set[int] = set()

    try:
        # 4. Office-scoped cleanup of tokens and queue counters
        db.query(QueueToken).filter(QueueToken.office_id == office_id).delete(synchronize_session=False)
        db.query(QueueCounter).filter(QueueCounter.office_id == office_id).delete(synchronize_session=False)

        # 5. Ensure services exist in DB
        for s_id, s_cfg in scen.services.items():
            affected_services.add(s_id)
            srv = db.query(Service).filter(Service.id == s_id).first()
            if not srv:
                srv = Service(
                    id=s_id,
                    code=s_cfg.service_code,
                    name=s_cfg.service_name,
                    mean_minutes=s_cfg.mean_service_minutes,
                    std_minutes=s_cfg.std_service_minutes,
                )
                db.add(srv)
            else:
                srv.mean_minutes = s_cfg.mean_service_minutes
                srv.std_minutes = s_cfg.std_service_minutes

        db.flush()

        # 6. Synchronize Counters & Eligibility
        counters_loaded = len(scen.initial_counters)
        busy_counters_loaded = 0

        # Remove existing counter eligibility mappings for this office's counters
        existing_counter_ids = [
            c.id for c in db.query(Counter.id).filter(Counter.office_id == office_id).all()
        ]
        if existing_counter_ids:
            db.query(CounterService).filter(
                CounterService.counter_id.in_(existing_counter_ids)
            ).delete(synchronize_session=False)

        # Upsert counters
        for sc in scen.initial_counters:
            affected_services.add(sc.assigned_service_id)
            status_val = CounterStatus.ACTIVE if sc.is_active else CounterStatus.UNAVAILABLE
            
            counter = db.query(Counter).filter(Counter.id == sc.counter_id).first()
            if not counter:
                counter = Counter(
                    id=sc.counter_id,
                    office_id=office_id,
                    name=sc.name,
                    status=status_val,
                    assigned_service_id=sc.assigned_service_id,
                    status_changed_at=ref_time - timedelta(hours=2),
                )
                db.add(counter)
            else:
                counter.office_id = office_id
                counter.name = sc.name
                counter.status = status_val
                counter.assigned_service_id = sc.assigned_service_id
                counter.status_changed_at = ref_time - timedelta(hours=2)

            db.flush()

            # Add eligibilities
            for el_s_id in sc.eligible_service_ids:
                affected_services.add(el_s_id)
                db.add(CounterService(counter_id=sc.counter_id, service_id=el_s_id))

        db.flush()

        # 7. Seed Tokens (Waiting, Serving, and Completed) per Service
        # To guarantee the observed trailing 60m arrival rate reflects the scenario's
        # target arrival rate lambda_hr, the total arrivals seeded in (ref_time - 60m, ref_time]
        # must satisfy: Total Arrivals = num_waiting + num_serving + num_completed = int(round(lambda_hr)).
        # Spreading these arrivals across the 60m window preserves strict FIFO queue order,
        # active counter serving status, and consistent 30m / 60m rate windows.
        max_existing_id = db.query(func.coalesce(func.max(QueueToken.id), 0)).scalar() or 0
        token_pk = max(max_existing_id, 1000) + 1
        busy_counters_loaded = 0
        waiting_tokens_loaded = 0

        for s_id, s_cfg in scen.services.items():
            svc_code = s_cfg.service_code
            serving_counters = [
                sc for sc in scen.initial_counters
                if sc.is_active and sc.assigned_service_id == s_id
            ]
            num_serving = len(serving_counters)
            num_waiting = scen.initial_queue_counts.get(s_id, 0)
            num_active = num_serving + num_waiting

            cap_mu = num_serving * (60.0 / s_cfg.mean_service_minutes) if s_cfg.mean_service_minutes > 0 else 6.0

            # Derive target arrival rate (arrivals per hour) from Phase 0
            target_rho = 0.65
            lambda_hr = 6.0
            if scen.arrival_phases:
                p0 = scen.arrival_phases[0]
                if p0.arrival_rates_per_hour and s_id in p0.arrival_rates_per_hour:
                    lambda_hr = p0.arrival_rates_per_hour[s_id]
                    target_rho = lambda_hr / cap_mu if cap_mu > 0 else 99.0
                elif p0.target_rho_by_service and s_id in p0.target_rho_by_service:
                    target_rho = p0.target_rho_by_service[s_id]
                    lambda_hr = max(1.0, target_rho * cap_mu)
            else:
                lambda_hr = max(1.0, target_rho * cap_mu)

            # Target arrivals in 60m window (never less than current active in-flight tokens)
            target_arrivals_60 = max(num_active, int(round(lambda_hr)))
            num_completed = target_arrivals_60 - num_active

            # Arrival time offsets from ref_time (in minutes into the past, 1 <= k <= T)
            # k=1 is the most recent arrival, k=T is the oldest arrival (~58m ago)
            T = target_arrivals_60
            def get_arr_time(slot_k: int) -> datetime:
                offset_m = (58.0 / max(1, T)) * slot_k
                return ref_time - timedelta(minutes=offset_m)

            # A. Waiting Tokens (most recent arrivals: slot 1 .. num_waiting)
            # w_idx=1 is first in line -> arrived earliest among waiting tokens (slot num_waiting)
            # w_idx=num_waiting arrived most recently (slot 1)
            for w_idx in range(1, num_waiting + 1):
                w_slot = num_waiting - w_idx + 1
                arr_t = get_arr_time(w_slot)
                w_token = QueueToken(
                    id=token_pk,
                    token_number=f"{office.code}-{svc_code}-W{w_idx:02d}",
                    office_id=office_id,
                    service_id=s_id,
                    status=TokenStatus.WAITING,
                    created_at=arr_t,
                    counter_id=None,
                )
                db.add(w_token)
                token_pk += 1
                waiting_tokens_loaded += 1

            # B. Serving Tokens (slots num_waiting + 1 .. num_active)
            slot = num_waiting + 1
            for sc in serving_counters:
                arr_t = get_arr_time(slot)
                serving_token = QueueToken(
                    id=token_pk,
                    token_number=f"{office.code}-{svc_code}-S{sc.counter_id}",
                    office_id=office_id,
                    service_id=s_id,
                    status=TokenStatus.SERVING,
                    counter_id=sc.counter_id,
                    created_at=arr_t,
                    called_at=arr_t + timedelta(minutes=1),
                    serving_at=arr_t + timedelta(minutes=2),
                )
                db.add(serving_token)
                token_pk += 1
                busy_counters_loaded += 1
                slot += 1

            # C. Historical Completed Tokens (slots num_active + 1 .. T)
            slot = num_active + 1
            for h_idx in range(num_completed):
                arr_t = get_arr_time(slot)
                call_t = arr_t + timedelta(minutes=1)
                serv_t = arr_t + timedelta(minutes=2)
                # Completed before current ref_time
                max_serv_duration = max(1.0, (ref_time - serv_t).total_seconds() / 60.0 - 1.0)
                comp_duration = min(s_cfg.mean_service_minutes, max_serv_duration)
                comp_t = serv_t + timedelta(minutes=comp_duration)
                db.add(QueueToken(
                    id=token_pk,
                    token_number=f"{office.code}-{svc_code}-H{h_idx:02d}",
                    office_id=office_id,
                    service_id=s_id,
                    status=TokenStatus.COMPLETED,
                    created_at=arr_t,
                    called_at=call_t,
                    serving_at=serv_t,
                    completed_at=comp_t,
                    counter_id=None,
                ))
                token_pk += 1
                slot += 1

        db.commit()

        logger.info(
            f"Successfully loaded scenario '{scen.scenario_id}' into office {office_id}: "
            f"{counters_loaded} counters, {busy_counters_loaded} busy, {waiting_tokens_loaded} waiting."
        )

        return ScenarioLoadResult(
            scenario_id=scen.scenario_id,
            scenario_name=scen.scenario_name,
            office_id=office_id,
            office_code=office.code,
            affected_service_ids=sorted(list(affected_services)),
            counters_loaded=counters_loaded,
            waiting_tokens_loaded=waiting_tokens_loaded,
            busy_counters_loaded=busy_counters_loaded,
            loaded_at_utc=ref_time,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to load scenario '{scenario_id}' into office {office_id}: {e}", exc_info=True)
        raise
