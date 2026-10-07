"""
PRAVAAH — Step 2D.4: Scenario Benchmarking & Calibration Engine.

Executes deterministic multi-replication benchmarks comparing:
1. Baseline (No Intervention)
2. Naive (Heuristic: Longest Queue / Lowest Traffic Intensity)
3. PRAVAAH (Simulation-Based Optimizer with P90 constraints, candidate durations, and honest no-action)

Guarantees:
- Common Random Numbers (CRN): Exact same seed sequence across all policies per replication.
- Unbiased evaluation: Naive and PRAVAAH share the same candidate/action space.
- Zero result manipulation: Honest reporting of whether PRAVAAH beats Naive or chooses Baseline.
- Detailed statistical reporting: Mean, standard deviation, min, max, P90, per-service breakdown.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from app.interventions.candidates import (
    build_naive_intervention,
    generate_candidate_interventions,
)
from app.interventions.evaluator import InterventionEvaluator
from app.interventions.metrics import (
    aggregate_replications,
    calculate_percentile,
)
from app.interventions.models import (
    EvaluatorConfig,
    EvaluationResult,
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)
from app.interventions.policies import (
    candidate_to_engine_interventions,
)
from app.simulation.engine import DiscreteEventEngine
from app.simulation.results import SimulationResult
from app.simulation.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioConfig,
    get_scenario,
)
from app.simulation.state import SimulationState


@dataclass(frozen=True)
class ServiceBenchmarkMetrics:
    """Aggregated per-service metrics across replications."""
    service_id: int
    service_code: str
    mean_total_waiting_person_minutes: float
    mean_avg_wait_minutes: float
    p90_wait_minutes: float
    mean_served: float
    mean_unserved: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_code": self.service_code,
            "mean_total_waiting_person_minutes": round(self.mean_total_waiting_person_minutes, 4),
            "mean_avg_wait_minutes": round(self.mean_avg_wait_minutes, 4),
            "p90_wait_minutes": round(self.p90_wait_minutes, 4),
            "mean_served": round(self.mean_served, 2),
            "mean_unserved": round(self.mean_unserved, 2),
        }


@dataclass(frozen=True)
class PolicyBenchmarkMetrics:
    """Aggregated performance metrics for a specific policy over multiple replications."""
    policy_type: PolicyType
    candidate_id: Optional[str]
    candidate_description: str
    replications: int
    mean_total_waiting_person_minutes: float
    std_total_waiting_person_minutes: float
    min_total_waiting_person_minutes: float
    max_total_waiting_person_minutes: float
    mean_avg_waiting_time_minutes: float
    p90_waiting_time_minutes: float
    overall_max_waiting_time_minutes: float
    mean_customers_served: float
    mean_unserved_count: float
    per_service_metrics: Dict[int, ServiceBenchmarkMetrics]
    replication_total_waits: Tuple[float, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_type": self.policy_type.value,
            "candidate_id": self.candidate_id,
            "candidate_description": self.candidate_description,
            "replications": self.replications,
            "mean_total_waiting_person_minutes": round(self.mean_total_waiting_person_minutes, 4),
            "std_total_waiting_person_minutes": round(self.std_total_waiting_person_minutes, 4),
            "min_total_waiting_person_minutes": round(self.min_total_waiting_person_minutes, 4),
            "max_total_waiting_person_minutes": round(self.max_total_waiting_person_minutes, 4),
            "mean_avg_waiting_time_minutes": round(self.mean_avg_waiting_time_minutes, 4),
            "p90_waiting_time_minutes": round(self.p90_waiting_time_minutes, 4),
            "overall_max_waiting_time_minutes": round(self.overall_max_waiting_time_minutes, 4),
            "mean_customers_served": round(self.mean_customers_served, 2),
            "mean_unserved_count": round(self.mean_unserved_count, 2),
            "per_service": {s_id: m.to_dict() for s_id, m in self.per_service_metrics.items()},
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmarking output for a scenario comparing Baseline, Naive, and PRAVAAH."""
    scenario_id: str
    scenario_name: str
    office_code: str
    horizon_minutes: float
    replications: int
    baseline: PolicyBenchmarkMetrics
    naive: PolicyBenchmarkMetrics
    pravaah: PolicyBenchmarkMetrics
    selected_policy: PolicyType
    selected_candidate_id: Optional[str]
    selected_action_description: str
    improvement_vs_baseline_person_min: float
    pct_improvement_vs_baseline: float
    improvement_vs_naive_person_min: float
    pct_improvement_vs_naive: float
    pravaah_beats_baseline: bool
    pravaah_beats_naive: bool
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "office_code": self.office_code,
            "horizon_minutes": self.horizon_minutes,
            "replications": self.replications,
            "baseline": self.baseline.to_dict(),
            "naive": self.naive.to_dict(),
            "pravaah": self.pravaah.to_dict(),
            "selected_policy": self.selected_policy.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_action_description": self.selected_action_description,
            "improvement_vs_baseline_person_min": round(self.improvement_vs_baseline_person_min, 4),
            "pct_improvement_vs_baseline": round(self.pct_improvement_vs_baseline, 2),
            "improvement_vs_naive_person_min": round(self.improvement_vs_naive_person_min, 4),
            "pct_improvement_vs_naive": round(self.pct_improvement_vs_naive, 2),
            "pravaah_beats_baseline": self.pravaah_beats_baseline,
            "pravaah_beats_naive": self.pravaah_beats_naive,
            "observations": self.observations,
        }


def _compute_policy_metrics(
    policy_type: PolicyType,
    candidate: Optional[InterventionCandidate],
    results: List[SimulationResult],
    services_map: Dict[int, Any],
) -> PolicyBenchmarkMetrics:
    """Computes comprehensive statistical metrics from multi-replication simulation results."""
    rep_count = len(results)
    if rep_count == 0:
        raise ValueError("Cannot compute metrics from 0 simulation results")

    total_waits = [r.total_waiting_person_minutes for r in results]
    mean_wait = sum(total_waits) / rep_count
    var_wait = sum((w - mean_wait) ** 2 for w in total_waits) / max(1, rep_count - 1)
    std_wait = math.sqrt(var_wait)
    min_wait = min(total_waits)
    max_wait = max(total_waits)

    avg_waits = [r.avg_waiting_time_minutes for r in results]
    mean_avg_wait = sum(avg_waits) / rep_count

    max_waits = [r.max_waiting_time_minutes for r in results]
    overall_max_wait = max(max_waits, default=0.0)

    served_counts = [r.total_customers_served for r in results]
    mean_served = sum(served_counts) / rep_count

    unserved_counts = [len(r.unserved_citizen_ids) for r in results]
    mean_unserved = sum(unserved_counts) / rep_count

    # Aggregate all individual wait times across all replications to calculate overall P90
    all_individual_waits: List[float] = []
    for r in results:
        for s_waits in r.service_waiting_times.values():
            all_individual_waits.extend(s_waits)
    overall_p90 = calculate_percentile(all_individual_waits, p=90.0)

    # Per-service breakdown
    service_metrics: Dict[int, ServiceBenchmarkMetrics] = {}
    for s_id, svc_cfg in services_map.items():
        s_total_waits: List[float] = []
        s_avg_waits: List[float] = []
        s_served: List[int] = []
        s_unserved: List[int] = []
        s_ind_waits: List[float] = []

        for r in results:
            if s_id in r.service_metrics:
                sm = r.service_metrics[s_id]
                s_total_waits.append(sm.total_waiting_person_minutes)
                s_avg_waits.append(sm.avg_waiting_time_minutes)
                s_served.append(sm.customers_served)
                s_unserved.append(sm.unserved_customers_count)
            if s_id in r.service_waiting_times:
                s_ind_waits.extend(r.service_waiting_times[s_id])

        service_metrics[s_id] = ServiceBenchmarkMetrics(
            service_id=s_id,
            service_code=svc_cfg.service_code,
            mean_total_waiting_person_minutes=sum(s_total_waits) / rep_count if s_total_waits else 0.0,
            mean_avg_wait_minutes=sum(s_avg_waits) / rep_count if s_avg_waits else 0.0,
            p90_wait_minutes=calculate_percentile(s_ind_waits, p=90.0),
            mean_served=sum(s_served) / rep_count if s_served else 0.0,
            mean_unserved=sum(s_unserved) / rep_count if s_unserved else 0.0,
        )

    cand_id = candidate.candidate_id if candidate else None
    cand_desc = candidate.description if candidate else "No counter reassignment (Baseline)"

    return PolicyBenchmarkMetrics(
        policy_type=policy_type,
        candidate_id=cand_id,
        candidate_description=cand_desc,
        replications=rep_count,
        mean_total_waiting_person_minutes=mean_wait,
        std_total_waiting_person_minutes=std_wait,
        min_total_waiting_person_minutes=min_wait,
        max_total_waiting_person_minutes=max_wait,
        mean_avg_waiting_time_minutes=mean_avg_wait,
        p90_waiting_time_minutes=overall_p90,
        overall_max_waiting_time_minutes=overall_max_wait,
        mean_customers_served=mean_served,
        mean_unserved_count=mean_unserved,
        per_service_metrics=service_metrics,
        replication_total_waits=tuple(total_waits),
    )


def run_scenario_benchmark(
    scenario: Union[str, ScenarioConfig],
    replications: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> BenchmarkResult:
    """
    Executes a benchmark comparison of Baseline, Naive, and PRAVAAH for a given scenario.
    Enforces Common Random Numbers (identical seeds) across all policies.
    """
    # 1. Resolve scenario config
    if isinstance(scenario, str):
        scen_config = get_scenario(scenario)
    elif isinstance(scenario, ScenarioConfig):
        scen_config = scenario
    else:
        raise TypeError(f"scenario must be str or ScenarioConfig (got {type(scenario)})")

    rep_count = replications if replications is not None else scen_config.experiment_params.replications
    seed_base = base_seed if base_seed is not None else scen_config.base_seed

    # 2. Build initial state
    initial_state = scen_config.build_simulation_state()

    # 3. Setup evaluator config wired to scenario factories and parameters
    eval_config = scen_config.build_evaluator_config()
    # Override replications and base_seed if explicitly passed
    eval_config = EvaluatorConfig(
        horizon_minutes=scen_config.horizon_minutes,
        replications=rep_count,
        base_seed=seed_base,
        p90_thresholds=eval_config.p90_thresholds,
        arrival_rates_per_hour=eval_config.arrival_rates_per_hour,
        candidate_durations=scen_config.experiment_params.candidate_durations_minutes,
        switching_delay_min=scen_config.experiment_params.switching_delay_minutes,
        availability_events=tuple(scen_config.availability_events),
        arrival_process_factory=lambda st, rng: scen_config.build_arrival_process(st, rng=rng),
        service_process_factory=lambda rng: scen_config.build_service_process(rng=rng),
    )

    evaluator = InterventionEvaluator(eval_config)
    eval_result = evaluator.evaluate(initial_state, config=eval_config)

    seeds = [seed_base + (rep * 1000) for rep in range(rep_count)]

    # 4. Simulate Baseline (No Intervention)
    baseline_results = evaluator._simulate_policy(initial_state, candidate=None, seeds=seeds)
    baseline_metrics = _compute_policy_metrics(
        policy_type=PolicyType.BASELINE,
        candidate=None,
        results=baseline_results,
        services_map=initial_state.services,
    )

    # 5. Simulate Naive Policy
    naive_candidate = build_naive_intervention(initial_state)
    naive_results = evaluator._simulate_policy(initial_state, candidate=naive_candidate, seeds=seeds)
    naive_metrics = _compute_policy_metrics(
        policy_type=PolicyType.NAIVE,
        candidate=naive_candidate,
        results=naive_results,
        services_map=initial_state.services,
    )

    # 6. Simulate PRAVAAH Selected Policy
    pravaah_candidate = eval_result.selected_candidate
    pravaah_policy_type = eval_result.selected_policy
    pravaah_results = evaluator._simulate_policy(initial_state, candidate=pravaah_candidate, seeds=seeds)
    pravaah_metrics = _compute_policy_metrics(
        policy_type=pravaah_policy_type,
        candidate=pravaah_candidate,
        results=pravaah_results,
        services_map=initial_state.services,
    )

    # 7. Comparison & Improvements
    base_wait = baseline_metrics.mean_total_waiting_person_minutes
    naive_wait = naive_metrics.mean_total_waiting_person_minutes
    prav_wait = pravaah_metrics.mean_total_waiting_person_minutes

    imp_vs_base = base_wait - prav_wait
    pct_imp_vs_base = (imp_vs_base / base_wait * 100.0) if base_wait > 1e-6 else 0.0

    imp_vs_naive = naive_wait - prav_wait
    pct_imp_vs_naive = (imp_vs_naive / naive_wait * 100.0) if naive_wait > 1e-6 else 0.0

    pravaah_beats_base = (prav_wait < base_wait - 1e-4)
    pravaah_beats_naive = (prav_wait < naive_wait - 1e-4)

    # 8. Observations
    observations: List[str] = []
    if pravaah_policy_type == PolicyType.BASELINE:
        observations.append("PRAVAAH correctly concluded NO INTERVENTION REQUIRED (Baseline is optimal).")
    elif pravaah_beats_naive:
        observations.append(
            f"PRAVAAH outperformed Naive by {imp_vs_naive:.2f} person-minutes ({pct_imp_vs_naive:.1f}% improvement)."
        )
    elif abs(imp_vs_naive) <= 1e-4:
        observations.append("PRAVAAH selected the Naive policy or an equivalent action.")
    else:
        observations.append(
            f"Naive achieved lower waiting person-minutes than selected candidate by {-imp_vs_naive:.2f} person-minutes."
        )

    if pravaah_candidate is not None:
        if pravaah_candidate.duration_min is not None:
            observations.append(
                f"Selected temporary intervention duration: {pravaah_candidate.duration_min:.0f} minutes "
                f"(reversion switching: {pravaah_candidate.switching_delay_min:.0f}m)."
            )
        else:
            observations.append("Selected permanent reassignment.")

    return BenchmarkResult(
        scenario_id=scen_config.scenario_id,
        scenario_name=scen_config.scenario_name,
        office_code=scen_config.office_code,
        horizon_minutes=scen_config.horizon_minutes,
        replications=rep_count,
        baseline=baseline_metrics,
        naive=naive_metrics,
        pravaah=pravaah_metrics,
        selected_policy=pravaah_policy_type,
        selected_candidate_id=pravaah_candidate.candidate_id if pravaah_candidate else None,
        selected_action_description=pravaah_metrics.candidate_description,
        improvement_vs_baseline_person_min=imp_vs_base,
        pct_improvement_vs_baseline=pct_imp_vs_base,
        improvement_vs_naive_person_min=imp_vs_naive,
        pct_improvement_vs_naive=pct_imp_vs_naive,
        pravaah_beats_baseline=pravaah_beats_base,
        pravaah_beats_naive=pravaah_beats_naive,
        observations=observations,
    )


def run_all_benchmarks(
    replications: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Executes benchmark comparison across all registered PRAVAAH scenarios.
    """
    results: Dict[str, BenchmarkResult] = {}
    for scen_id in sorted(SCENARIO_REGISTRY.keys()):
        results[scen_id] = run_scenario_benchmark(
            scen_id,
            replications=replications,
            base_seed=base_seed,
        )
    return results
