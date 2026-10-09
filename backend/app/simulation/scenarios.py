"""
PRAVAAH — Step 2D.2 Simulation Scenarios & Domain Data Configuration.

Provides realistic, deterministic, reproducible scenario fixtures and domain configurations
for discrete-event simulation benchmarking:
- Healthy operation (baseline optimal, no intervention)
- Queue pressure (Property Tax backlog, candidate donor optimization)
- Donor-service constraints (protecting near-capacity donor queues)
- Temporary intervention / look-ahead burst (reversion-duration experiment representation)
- Counter failure / congestion (capacity reduction and queue stability)

Architectural Guarantees:
- Zero database or SQLAlchemy dependencies.
- Pure Python dataclass structures.
- Isolated, seed-driven RNG (no global random mutation).
- Capacity mu is dynamically derived from service mean times and active counter assignments.
- Non-negative Lognormal service times with CV = 0.40.
- Decoupled from HTTP, DB, and ML layers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from app.simulation.engine import CounterAvailabilityEvent
from app.simulation.processes import (
    ArrivalBurst,
    ArrivalPhase,
    LognormalServiceProcess,
    PhaseArrivalProcess,
)
from app.simulation.state import (
    SimCitizen,
    SimCounter,
    SimQueue,
    SimulationState,
)
from shared.features import ServiceConfigSnapshot


# =============================================================================
# 1. LOCKED DOMAIN IDENTIFIERS & CONSTANTS
# =============================================================================

# Offices
OFFICE_MUNICIPALITY_ID = 1
OFFICE_MUNICIPALITY_CODE = "MUNI"
OFFICE_MUNICIPALITY_NAME = "Municipality"

OFFICE_RTO_ID = 2
OFFICE_RTO_CODE = "RTO"
OFFICE_RTO_NAME = "Regional Transport Office"

# Municipality Services (PT, WC, BC)
SERVICE_PT_ID = 1
SERVICE_PT_CODE = "PT"
SERVICE_PT_NAME = "Property Tax"
SERVICE_PT_MEAN = 10.0  # minutes

SERVICE_WC_ID = 2
SERVICE_WC_CODE = "WC"
SERVICE_WC_NAME = "Water Service"
SERVICE_WC_MEAN = 15.0  # minutes

SERVICE_BC_ID = 3
SERVICE_BC_CODE = "BC"
SERVICE_BC_NAME = "Birth Certificate"
SERVICE_BC_MEAN = 7.0   # minutes

# RTO Services (OT, DR)
SERVICE_OT_ID = 4
SERVICE_OT_CODE = "OT"
SERVICE_OT_NAME = "Vehicle Ownership Transfer"
SERVICE_OT_MEAN = 14.0  # minutes

SERVICE_DR_ID = 5
SERVICE_DR_CODE = "DR"
SERVICE_DR_NAME = "Duplicate/Replacement RC"
SERVICE_DR_MEAN = 8.0   # minutes

# Global initial coefficient of variation across all services
DEFAULT_CV = 0.40


# =============================================================================
# 2. STANDARD SERVICE & COUNTER DOMAIN DEFINITIONS
# =============================================================================

def get_standard_municipality_services() -> Dict[int, ServiceConfigSnapshot]:
    """
    Returns standard Municipality service configuration snapshots (PT, WC, BC).
    Strictly excludes Income Certificate.
    """
    return {
        SERVICE_PT_ID: ServiceConfigSnapshot(
            service_id=SERVICE_PT_ID,
            service_code=SERVICE_PT_CODE,
            service_name=SERVICE_PT_NAME,
            mean_service_minutes=SERVICE_PT_MEAN,
            std_service_minutes=SERVICE_PT_MEAN * DEFAULT_CV,
        ),
        SERVICE_WC_ID: ServiceConfigSnapshot(
            service_id=SERVICE_WC_ID,
            service_code=SERVICE_WC_CODE,
            service_name=SERVICE_WC_NAME,
            mean_service_minutes=SERVICE_WC_MEAN,
            std_service_minutes=SERVICE_WC_MEAN * DEFAULT_CV,
        ),
        SERVICE_BC_ID: ServiceConfigSnapshot(
            service_id=SERVICE_BC_ID,
            service_code=SERVICE_BC_CODE,
            service_name=SERVICE_BC_NAME,
            mean_service_minutes=SERVICE_BC_MEAN,
            std_service_minutes=SERVICE_BC_MEAN * DEFAULT_CV,
        ),
    }


def get_standard_rto_services() -> Dict[int, ServiceConfigSnapshot]:
    """
    Returns standard RTO service configuration snapshots (OT, DR).
    """
    return {
        SERVICE_OT_ID: ServiceConfigSnapshot(
            service_id=SERVICE_OT_ID,
            service_code=SERVICE_OT_CODE,
            service_name=SERVICE_OT_NAME,
            mean_service_minutes=SERVICE_OT_MEAN,
            std_service_minutes=SERVICE_OT_MEAN * DEFAULT_CV,
        ),
        SERVICE_DR_ID: ServiceConfigSnapshot(
            service_id=SERVICE_DR_ID,
            service_code=SERVICE_DR_CODE,
            service_name=SERVICE_DR_NAME,
            mean_service_minutes=SERVICE_DR_MEAN,
            std_service_minutes=SERVICE_DR_MEAN * DEFAULT_CV,
        ),
    }


def get_standard_municipality_counters() -> List[SimCounter]:
    """
    Returns standard physical counters C1 through C6 for the Municipality.
    Eligibilities:
      C1: home PT, eligible {PT, BC}
      C2: home PT, eligible {PT, WC}
      C3: home WC, eligible {WC, PT}
      C4: home WC, eligible {WC}
      C5: home BC, eligible {BC, PT}
      C6: home BC, eligible {BC}
    Normal assignment:
      C1 -> PT, C2 -> PT, C3 -> WC, C4 -> WC, C5 -> BC, C6 -> BC
    """
    return [
        SimCounter(
            counter_id=1,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C1",
            assigned_service_id=SERVICE_PT_ID,
            eligible_service_ids=frozenset({SERVICE_PT_ID, SERVICE_BC_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=2,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C2",
            assigned_service_id=SERVICE_PT_ID,
            eligible_service_ids=frozenset({SERVICE_PT_ID, SERVICE_WC_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=3,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C3",
            assigned_service_id=SERVICE_WC_ID,
            eligible_service_ids=frozenset({SERVICE_WC_ID, SERVICE_PT_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=4,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C4",
            assigned_service_id=SERVICE_WC_ID,
            eligible_service_ids=frozenset({SERVICE_WC_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=5,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C5",
            assigned_service_id=SERVICE_BC_ID,
            eligible_service_ids=frozenset({SERVICE_BC_ID, SERVICE_PT_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=6,
            office_id=OFFICE_MUNICIPALITY_ID,
            name="C6",
            assigned_service_id=SERVICE_BC_ID,
            eligible_service_ids=frozenset({SERVICE_BC_ID}),
            is_active=True,
        ),
    ]


def get_standard_rto_counters() -> List[SimCounter]:
    """
    Returns standard physical counters R1 through R3 for the RTO.
    Eligibilities:
      R1: home OT, eligible {OT}
      R2: home OT, eligible {OT, DR}
      R3: home DR, eligible {DR}
    Normal assignment:
      R1 -> OT, R2 -> OT, R3 -> DR
    """
    return [
        SimCounter(
            counter_id=201,
            office_id=OFFICE_RTO_ID,
            name="R1",
            assigned_service_id=SERVICE_OT_ID,
            eligible_service_ids=frozenset({SERVICE_OT_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=202,
            office_id=OFFICE_RTO_ID,
            name="R2",
            assigned_service_id=SERVICE_OT_ID,
            eligible_service_ids=frozenset({SERVICE_OT_ID, SERVICE_DR_ID}),
            is_active=True,
        ),
        SimCounter(
            counter_id=203,
            office_id=OFFICE_RTO_ID,
            name="R3",
            assigned_service_id=SERVICE_DR_ID,
            eligible_service_ids=frozenset({SERVICE_DR_ID}),
            is_active=True,
        ),
    ]


# =============================================================================
# 3. CAPACITY CALCULATIONS
# =============================================================================

def compute_counter_capacity_mu(mean_service_minutes: float) -> float:
    """
    Derives the service capacity mu (citizens/hour) for a single counter:
      mu = 60.0 / mean_service_minutes

    Examples:
      PT (10 min) -> 6.00 / hr
      WC (15 min) -> 4.00 / hr
      BC (7 min)  -> 8.57 / hr
      OT (14 min) -> 4.29 / hr
      DR (8 min)  -> 7.50 / hr
    """
    if mean_service_minutes <= 0.0:
        raise ValueError(f"mean_service_minutes must be > 0.0 (got {mean_service_minutes})")
    return 60.0 / mean_service_minutes


def compute_service_capacity_mu(state: SimulationState, service_id: int) -> float:
    """
    Derives service capacity mu (citizens/hour) directly from the simulation state:
      mu_service = N_active_assigned * (60.0 / mean_service_minutes)

    Guarantees:
    - Automatically updates when counters are reassigned.
    - Automatically updates when counters become UNAVAILABLE or AVAILABLE.
    - Zero separate/conflicting capacity calculations.
    """
    svc_cfg = state.services.get(service_id)
    if not svc_cfg or svc_cfg.mean_service_minutes <= 0.0:
        return 0.0

    active_assigned_count = sum(
        1 for c in state.counters.values()
        if c.is_active and c.assigned_service_id == service_id
    )
    if active_assigned_count == 0:
        return 0.0

    return active_assigned_count * (60.0 / svc_cfg.mean_service_minutes)


def compute_all_service_capacities_mu(state: SimulationState) -> Dict[int, float]:
    """
    Computes capacity mu (citizens/hour) for every service present in state.
    """
    return {
        s_id: compute_service_capacity_mu(state, s_id)
        for s_id in state.services.keys()
    }


# =============================================================================
# 4. SCENARIO CONFIGURATION STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class InterventionExperimentParams:
    """
    Configuration parameters for intervention evaluation experiments.
    Note: p90_thresholds are uncalibrated exploration bounds; final calibration
    is deferred until empirical distributions are observed.
    """
    switching_delay_minutes: float = 3.0
    candidate_durations_minutes: Tuple[float, ...] = (15.0, 30.0)
    replications: int = 30
    p90_thresholds: Dict[int, Optional[float]] = field(default_factory=dict)
    # Execution note: Single-reassignment is supported by engine.
    # Reversion after 15/30 min duration is configuration-ready and execution-deferred.
    reversion_execution_supported: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Complete typed specification of a discrete-event simulation scenario.
    """
    scenario_id: str
    scenario_name: str
    description: str
    office_id: int
    office_code: str
    horizon_minutes: float = 60.0
    base_seed: int = 42
    services: Dict[int, ServiceConfigSnapshot] = field(default_factory=dict)
    initial_counters: List[SimCounter] = field(default_factory=list)
    initial_queue_counts: Dict[int, int] = field(default_factory=dict)
    arrival_phases: List[ArrivalPhase] = field(default_factory=list)
    arrival_bursts: List[ArrivalBurst] = field(default_factory=list)
    availability_events: List[CounterAvailabilityEvent] = field(default_factory=list)
    experiment_params: InterventionExperimentParams = field(
        default_factory=InterventionExperimentParams
    )

    def build_simulation_state(
        self,
        snapshot_time_utc: Optional[datetime] = None,
    ) -> SimulationState:
        """
        Builds a pristine, isolated SimulationState from this scenario configuration.
        Zero database access or global state mutation.
        """
        t_snap = snapshot_time_utc or datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)
        services_map = dict(self.services)
        counters_map = {c.counter_id: c.clone() for c in self.initial_counters}

        # Build initial queues
        queues_map: Dict[int, SimQueue] = {}
        token_counter = 100

        for s_id in self.services.keys():
            waiting_citizens: List[SimCitizen] = []
            count = self.initial_queue_counts.get(s_id, 0)
            svc_code = self.services[s_id].service_code
            for i in range(count):
                token_counter += 1
                # Spread initial arrivals before snapshot start
                arr_offset = -float((count - i) * 2.0)
                waiting_citizens.append(
                    SimCitizen(
                        token_id=token_counter,
                        token_number=f"{svc_code}-{token_counter}",
                        service_id=s_id,
                        arrival_time_min=arr_offset,
                    )
                )
            queues_map[s_id] = SimQueue(service_id=s_id, waiting_citizens=waiting_citizens)

        # Derive initial traffic intensity rho by service from Phase 0
        rho_by_service: Dict[int, float] = {}
        if self.arrival_phases:
            p0 = self.arrival_phases[0]
            if p0.target_rho_by_service:
                rho_by_service = dict(p0.target_rho_by_service)
            elif p0.arrival_rates_per_hour:
                for s_id, rate in p0.arrival_rates_per_hour.items():
                    svc = services_map.get(s_id)
                    if svc and svc.mean_service_minutes > 0.0:
                        active_count = sum(
                            1 for c in counters_map.values()
                            if c.is_active and c.assigned_service_id == s_id
                        )
                        mu = active_count * (60.0 / svc.mean_service_minutes)
                        rho_by_service[s_id] = rate / mu if mu > 0.0 else 99.0

        return SimulationState(
            office_id=self.office_id,
            office_code=self.office_code,
            snapshot_time_utc=t_snap,
            services=services_map,
            queues=queues_map,
            counters=counters_map,
            initial_serving_tokens={},
            traffic_intensity_by_service=rho_by_service,
        )

    def build_arrival_process(
        self,
        state: SimulationState,
        rng: Optional[random.Random] = None,
        for_decision: bool = False,
    ) -> PhaseArrivalProcess:
        """
        Builds a PhaseArrivalProcess wired to the state's capacities and scenario phases.
        If for_decision is True, filters out unannounced arrival bursts.
        """
        capacities = compute_all_service_capacities_mu(state)
        bursts = [b for b in self.arrival_bursts if b.announced] if for_decision else self.arrival_bursts
        return PhaseArrivalProcess(
            phases=self.arrival_phases,
            service_capacities_per_hour=capacities,
            bursts=bursts,
            rng=rng if rng is not None else random.Random(self.base_seed),
        )

    def build_service_process(
        self,
        rng: Optional[random.Random] = None,
    ) -> LognormalServiceProcess:
        """
        Builds a LognormalServiceProcess using isolated seeded randomness.
        """
        return LognormalServiceProcess(
            rng=rng if rng is not None else random.Random(self.base_seed),
            default_cv=DEFAULT_CV,
        )

    def build_evaluator_config(self, for_decision: bool = True) -> "EvaluatorConfig":
        """
        Derives an EvaluatorConfig matching this scenario's arrival rates and parameters.
        Enforces information boundary: when for_decision=True, only announced events are visible.
        """
        from app.interventions.models import EvaluatorConfig

        rates: Dict[int, float] = {}
        if self.arrival_phases:
            p0 = self.arrival_phases[0]
            if p0.arrival_rates_per_hour:
                rates = dict(p0.arrival_rates_per_hour)
            elif p0.target_rho_by_service:
                state = self.build_simulation_state()
                caps = compute_all_service_capacities_mu(state)
                rates = {
                    s_id: rho * caps.get(s_id, 0.0)
                    for s_id, rho in p0.target_rho_by_service.items()
                }

        safe_p90 = {
            s_id: thresh
            for s_id, thresh in self.experiment_params.p90_thresholds.items()
            if thresh is not None
        }

        avail_events = (
            tuple(e for e in self.availability_events if e.announced)
            if for_decision
            else tuple(self.availability_events)
        )

        return EvaluatorConfig(
            horizon_minutes=self.horizon_minutes,
            replications=self.experiment_params.replications,
            base_seed=self.base_seed,
            p90_thresholds=safe_p90,
            arrival_rates_per_hour=rates if rates else None,
            candidate_durations=self.experiment_params.candidate_durations_minutes,
            switching_delay_min=self.experiment_params.switching_delay_minutes,
            availability_events=avail_events,
            arrival_process_factory=lambda st, rng: self.build_arrival_process(st, rng=rng, for_decision=for_decision),
            service_process_factory=lambda rng: self.build_service_process(rng=rng),
        )


# =============================================================================
# 5. FIVE STANDARD SCENARIO FIXTURES
# =============================================================================

def build_scenario_a_healthy() -> ScenarioConfig:
    """
    SCENARIO A — HEALTHY
    Purpose: Demonstrate that PRAVAAH correctly decides NO INTERVENTION REQUIRED.
    Conditions:
    - Balanced, stable queues across all Municipality services.
    - Target rho ~ 0.60–0.65 (healthy).
    - Light initial backlog (1 token per queue).
    """
    return ScenarioConfig(
        scenario_id="scenario_a_healthy",
        scenario_name="Scenario A: Healthy Operation",
        description=(
            "Balanced queues across Property Tax, Water Service, and Birth Certificate. "
            "Traffic intensity is healthy (rho ~ 0.60–0.65). "
            "Simulator must conclude that Baseline (no intervention) is optimal."
        ),
        office_id=OFFICE_MUNICIPALITY_ID,
        office_code=OFFICE_MUNICIPALITY_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_municipality_services(),
        initial_counters=get_standard_municipality_counters(),
        initial_queue_counts={
            SERVICE_PT_ID: 1,
            SERVICE_WC_ID: 1,
            SERVICE_BC_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="healthy_baseline",
                start_min=0.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 0.65,  # lambda = 0.65 * 12.0 = 7.80 / hr
                    SERVICE_WC_ID: 0.65,  # lambda = 0.65 * 8.0  = 5.20 / hr
                    SERVICE_BC_ID: 0.60,  # lambda = 0.60 * 17.14 = 10.28 / hr
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
        ),
    )


def build_scenario_b_pt_pressure() -> ScenarioConfig:
    """
    SCENARIO B — PT PRESSURE
    Purpose: Property Tax experiences significant pressure, while WC and BC remain normal.
    Conditions:
    - Initial queue of 8 tokens waiting for PT.
    - PT target rho ~ 1.05 (arrival rate exceeds capacity).
    - WC target rho = 0.60, BC target rho = 0.50 (both healthy).
    - Legitimate opportunity for candidate intervention (C3->PT or C5->PT).
    """
    return ScenarioConfig(
        scenario_id="scenario_b_pt_pressure",
        scenario_name="Scenario B: Property Tax Queue Pressure",
        description=(
            "Property Tax experiences elevated arrivals (rho = 1.05) and an existing backlog (8 tokens), "
            "while Water and Birth Certificate remain healthy. Creates opportunity for counter reassignment."
        ),
        office_id=OFFICE_MUNICIPALITY_ID,
        office_code=OFFICE_MUNICIPALITY_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_municipality_services(),
        initial_counters=get_standard_municipality_counters(),
        initial_queue_counts={
            SERVICE_PT_ID: 8,
            SERVICE_WC_ID: 1,
            SERVICE_BC_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="pt_pressure_phase",
                start_min=0.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 1.05,  # lambda = 1.05 * 12.0 = 12.60 / hr
                    SERVICE_WC_ID: 0.60,  # lambda = 0.60 * 8.0  = 4.80 / hr
                    SERVICE_BC_ID: 0.50,  # lambda = 0.50 * 17.14 = 8.57 / hr
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
        ),
    )


def build_scenario_c_donor_constraint() -> ScenarioConfig:
    """
    SCENARIO C — DONOR CONSTRAINT
    Purpose: Test 'Do not blindly steal capacity from another queue.'
    Conditions:
    - PT is under pressure (rho = 1.05, 8 waiting tokens).
    - WC is approaching its safe operating limit (rho = 0.90, 4 waiting tokens).
    - BC has spare headroom (rho = 0.45).
    - Stealing C3 (from WC) drops WC capacity from 8/hr to 4/hr, causing WC rho to jump to 1.80 (severe disruption).
    - Reassigning C5 (from BC) is viable because BC has ample capacity.
    """
    return ScenarioConfig(
        scenario_id="scenario_c_donor_constraint",
        scenario_name="Scenario C: Donor Capacity Constraint",
        description=(
            "PT requires assistance, but WC is near its operating limit (rho = 0.90). "
            "Reassigning C3 from WC would severely overload Water Service. "
            "Reassigning C5 from Birth Certificate (rho = 0.45) protects donor stability."
        ),
        office_id=OFFICE_MUNICIPALITY_ID,
        office_code=OFFICE_MUNICIPALITY_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_municipality_services(),
        initial_counters=get_standard_municipality_counters(),
        initial_queue_counts={
            SERVICE_PT_ID: 8,
            SERVICE_WC_ID: 4,
            SERVICE_BC_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="donor_constrained_phase",
                start_min=0.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 1.05,  # lambda = 12.60 / hr (capacity 12.0)
                    SERVICE_WC_ID: 0.90,  # lambda = 7.20 / hr (capacity 8.0)
                    SERVICE_BC_ID: 0.45,  # lambda = 7.71 / hr (capacity 17.14)
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
        ),
    )


def build_scenario_d_temporary_intervention() -> ScenarioConfig:
    """
    SCENARIO D — TEMPORARY INTERVENTION / LOOK-AHEAD
    Purpose: PT needs assistance now, but BC has an upcoming burst.
    Conditions:
    - Phase 1 (0 to 30 min): PT pressure (rho = 1.00), BC light (rho = 0.45).
    - Phase 2 (30 to 60 min): BC arrival burst (multiplier = 2.0, surging BC arrivals to 20.57/hr).
    - Allows benchmarking: C5 -> PT for 15 min vs 30 min.
    Execution Note: Reversion execution is formally deferred until multi-event
    reassignment scheduling is added in the next step.
    """
    return ScenarioConfig(
        scenario_id="scenario_d_temporary_intervention",
        scenario_name="Scenario D: Temporary Assistance and Look-Ahead Burst",
        description=(
            "PT needs temporary assistance. BC has current spare capacity, but a future burst "
            "occurs from t=30 to t=60. Benchmarks 15-minute vs 30-minute intervention durations. "
            "Reversion execution is configuration-ready and execution-deferred."
        ),
        office_id=OFFICE_MUNICIPALITY_ID,
        office_code=OFFICE_MUNICIPALITY_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_municipality_services(),
        initial_counters=get_standard_municipality_counters(),
        initial_queue_counts={
            SERVICE_PT_ID: 6,
            SERVICE_WC_ID: 1,
            SERVICE_BC_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="pre_burst_phase",
                start_min=0.0,
                end_min=30.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 1.00,
                    SERVICE_WC_ID: 0.60,
                    SERVICE_BC_ID: 0.45,
                },
            ),
            ArrivalPhase(
                phase_name="post_burst_phase",
                start_min=30.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 0.75,
                    SERVICE_WC_ID: 0.60,
                    SERVICE_BC_ID: 0.60,
                },
            ),
        ],
        arrival_bursts=[
            ArrivalBurst(
                burst_name="bc_surge_burst",
                service_id=SERVICE_BC_ID,
                start_min=30.0,
                end_min=60.0,
                multiplier=2.0,  # Surges BC arrival rate by 2.0x during 30-60 min
            )
        ],
        availability_events=[],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
            reversion_execution_supported=True,
        ),
    )


def build_scenario_e_counter_failure() -> ScenarioConfig:
    """
    SCENARIO E — COUNTER FAILURE
    Purpose: Simulate a physical counter becoming UNAVAILABLE.
    Conditions:
    - Counter C1 suffers a failure at t = 10.0 minutes (is_available=False).
    - Reduces PT capacity from 12.0/hr to 6.0/hr.
    - Demonstrates queue accumulation and congestion resulting from counter outage.
    """
    return ScenarioConfig(
        scenario_id="scenario_e_counter_failure",
        scenario_name="Scenario E: Counter Failure / Congestion",
        description=(
            "Counter C1 becomes UNAVAILABLE at t=10.0 min. Property Tax capacity is halved, "
            "causing effective traffic intensity to exceed 1.40 and creating severe queue buildup."
        ),
        office_id=OFFICE_MUNICIPALITY_ID,
        office_code=OFFICE_MUNICIPALITY_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_municipality_services(),
        initial_counters=get_standard_municipality_counters(),
        initial_queue_counts={
            SERVICE_PT_ID: 2,
            SERVICE_WC_ID: 1,
            SERVICE_BC_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="counter_failure_phase",
                start_min=0.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_PT_ID: 0.75,  # lambda = 9.00 / hr
                    SERVICE_WC_ID: 0.65,  # lambda = 5.20 / hr
                    SERVICE_BC_ID: 0.60,  # lambda = 10.28 / hr
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[
            CounterAvailabilityEvent(
                time_min=10.0,
                counter_id=1,
                is_available=False,
            )
        ],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
        ),
    )


def build_rto_scenario_healthy() -> ScenarioConfig:
    """
    Standard healthy scenario for the Regional Transport Office (RTO).
    Services: OT (mean=14m), DR (mean=8m).
    Counters: R1 (OT), R2 (OT), R3 (DR).
    """
    return ScenarioConfig(
        scenario_id="scenario_rto_healthy",
        scenario_name="RTO: Healthy Operation",
        description="Standard healthy operation for Vehicle Ownership Transfer and Duplicate RC.",
        office_id=OFFICE_RTO_ID,
        office_code=OFFICE_RTO_CODE,
        horizon_minutes=60.0,
        base_seed=42,
        services=get_standard_rto_services(),
        initial_counters=get_standard_rto_counters(),
        initial_queue_counts={
            SERVICE_OT_ID: 1,
            SERVICE_DR_ID: 1,
        },
        arrival_phases=[
            ArrivalPhase(
                phase_name="rto_healthy_phase",
                start_min=0.0,
                end_min=60.0,
                target_rho_by_service={
                    SERVICE_OT_ID: 0.65,  # 2 counters -> mu = 2 * (60/14) = 8.57 -> lambda = 5.57
                    SERVICE_DR_ID: 0.65,  # 1 counter -> mu = 1 * (60/8) = 7.50 -> lambda = 4.88
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[],
        experiment_params=InterventionExperimentParams(
            switching_delay_minutes=3.0,
            candidate_durations_minutes=(15.0, 30.0),
            replications=30,
        ),
    )


def build_scenario_d_announced() -> ScenarioConfig:
    """
    SCENARIO D (ANNOUNCED)
    Variant of Scenario D where the BC arrival surge is announced in advance (announced=True).
    PRAVAAH decision world is aware of the future surge from t=30 to t=60.
    """
    base_d = build_scenario_d_temporary_intervention()
    announced_bursts = [
        ArrivalBurst(
            burst_name=b.burst_name,
            service_id=b.service_id,
            start_min=b.start_min,
            end_min=b.end_min,
            multiplier=b.multiplier,
            additional_rate_per_hour=b.additional_rate_per_hour,
            announced=True,
        )
        for b in base_d.arrival_bursts
    ]
    return ScenarioConfig(
        scenario_id="scenario_d_announced",
        scenario_name="Scenario D (Announced): Announced Look-Ahead Burst",
        description="Scenario D with announced=True: future BC arrival surge is known to decision simulation.",
        office_id=base_d.office_id,
        office_code=base_d.office_code,
        horizon_minutes=base_d.horizon_minutes,
        base_seed=base_d.base_seed,
        services=base_d.services,
        initial_counters=base_d.initial_counters,
        initial_queue_counts=base_d.initial_queue_counts,
        arrival_phases=base_d.arrival_phases,
        arrival_bursts=announced_bursts,
        availability_events=base_d.availability_events,
        experiment_params=base_d.experiment_params,
    )


def build_scenario_d_unannounced() -> ScenarioConfig:
    """
    SCENARIO D (UNANNOUNCED)
    Variant of Scenario D where the BC arrival surge exists but is NOT announced (announced=False).
    PRAVAAH decision world simulates base traffic only; scoring world experiences the actual surge.
    """
    base_d = build_scenario_d_temporary_intervention()
    unannounced_bursts = [
        ArrivalBurst(
            burst_name=b.burst_name,
            service_id=b.service_id,
            start_min=b.start_min,
            end_min=b.end_min,
            multiplier=b.multiplier,
            additional_rate_per_hour=b.additional_rate_per_hour,
            announced=False,
        )
        for b in base_d.arrival_bursts
    ]
    return ScenarioConfig(
        scenario_id="scenario_d_unannounced",
        scenario_name="Scenario D (Unannounced): Unannounced Look-Ahead Burst",
        description="Scenario D with announced=False: future BC arrival surge is hidden from decision simulation.",
        office_id=base_d.office_id,
        office_code=base_d.office_code,
        horizon_minutes=base_d.horizon_minutes,
        base_seed=base_d.base_seed,
        services=base_d.services,
        initial_counters=base_d.initial_counters,
        initial_queue_counts=base_d.initial_queue_counts,
        arrival_phases=base_d.arrival_phases,
        arrival_bursts=unannounced_bursts,
        availability_events=base_d.availability_events,
        experiment_params=base_d.experiment_params,
    )


def build_scenario_e_known() -> ScenarioConfig:
    """
    SCENARIO E (KNOWN)
    Variant of Scenario E where counter C1 is already UNAVAILABLE at t=0 (part of current observable state).
    PRAVAAH decision world immediately observes the failure and reduced capacity at snapshot time.
    """
    base_e = build_scenario_e_counter_failure()
    # At t=0, C1 is already inactive
    known_counters = [
        SimCounter(
            counter_id=c.counter_id,
            office_id=c.office_id,
            name=c.name,
            assigned_service_id=c.assigned_service_id,
            eligible_service_ids=c.eligible_service_ids,
            is_active=False if c.counter_id == 1 else c.is_active,
        )
        for c in base_e.initial_counters
    ]
    return ScenarioConfig(
        scenario_id="scenario_e_known",
        scenario_name="Scenario E (Known): Known Counter Failure at t=0",
        description="Counter C1 failure is already active at t=0 (observable in initial state; no future event).",
        office_id=base_e.office_id,
        office_code=base_e.office_code,
        horizon_minutes=base_e.horizon_minutes,
        base_seed=base_e.base_seed,
        services=base_e.services,
        initial_counters=known_counters,
        initial_queue_counts=base_e.initial_queue_counts,
        arrival_phases=[
            ArrivalPhase(
                phase_name="counter_failure_known_phase",
                start_min=0.0,
                end_min=60.0,
                arrival_rates_per_hour={
                    SERVICE_PT_ID: 9.00,
                    SERVICE_WC_ID: 5.20,
                    SERVICE_BC_ID: 10.285714,
                },
            )
        ],
        arrival_bursts=[],
        availability_events=[],
        experiment_params=base_e.experiment_params,
    )


def build_scenario_e_unknown() -> ScenarioConfig:
    """
    SCENARIO E (UNKNOWN)
    Variant of Scenario E where counter C1 failure occurs at t=10.0 min and is UNANNOUNCED (announced=False).
    PRAVAAH decision world does not know of the outage; failure still executes during scoring.
    """
    base_e = build_scenario_e_counter_failure()
    unannounced_events = [
        CounterAvailabilityEvent(
            time_min=e.time_min,
            counter_id=e.counter_id,
            is_available=e.is_available,
            announced=False,
        )
        for e in base_e.availability_events
    ]
    return ScenarioConfig(
        scenario_id="scenario_e_unknown",
        scenario_name="Scenario E (Unknown): Unannounced Counter Failure at t=10",
        description="Counter C1 failure at t=10 is unannounced: hidden from decision world, active during scoring.",
        office_id=base_e.office_id,
        office_code=base_e.office_code,
        horizon_minutes=base_e.horizon_minutes,
        base_seed=base_e.base_seed,
        services=base_e.services,
        initial_counters=base_e.initial_counters,
        initial_queue_counts=base_e.initial_queue_counts,
        arrival_phases=base_e.arrival_phases,
        arrival_bursts=base_e.arrival_bursts,
        availability_events=unannounced_events,
        experiment_params=base_e.experiment_params,
    )


# =============================================================================
# 6. SCENARIO REGISTRY
# =============================================================================

SCENARIO_REGISTRY: Dict[str, Callable[[], ScenarioConfig]] = {
    # 6 Core Scenarios
    "scenario_a_healthy": build_scenario_a_healthy,
    "scenario_b_pt_pressure": build_scenario_b_pt_pressure,
    "scenario_c_donor_constraint": build_scenario_c_donor_constraint,
    "scenario_d_temporary_intervention": build_scenario_d_temporary_intervention,
    "scenario_d_look_ahead_burst": build_scenario_d_temporary_intervention,
    "scenario_e_counter_failure": build_scenario_e_counter_failure,
    "scenario_rto_healthy": build_rto_scenario_healthy,
    # 4 Information-Boundary Variants
    "scenario_d_announced": build_scenario_d_announced,
    "scenario_d_unannounced": build_scenario_d_unannounced,
    "scenario_e_known": build_scenario_e_known,
    "scenario_e_unknown": build_scenario_e_unknown,
    # Short alias keys
    "A": build_scenario_a_healthy,
    "B": build_scenario_b_pt_pressure,
    "C": build_scenario_c_donor_constraint,
    "D": build_scenario_d_temporary_intervention,
    "E": build_scenario_e_counter_failure,
    "RTO Healthy": build_rto_scenario_healthy,
    "D_announced": build_scenario_d_announced,
    "D_unannounced": build_scenario_d_unannounced,
    "E_known": build_scenario_e_known,
    "E_unknown": build_scenario_e_unknown,
}


def get_scenario(scenario_id: str) -> ScenarioConfig:
    """
    Factory function to retrieve a fresh, unmutated ScenarioConfig by identifier.
    """
    if scenario_id not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Scenario '{scenario_id}' not found in SCENARIO_REGISTRY "
            f"(available: {list(SCENARIO_REGISTRY.keys())})"
        )
    return SCENARIO_REGISTRY[scenario_id]()
