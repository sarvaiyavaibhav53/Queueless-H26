"""
PRAVAAH — Step 2D.4: Scenario Benchmarking, Calibration & Methodological Verification Engine.

Executes deterministic multi-replication benchmarks comparing:
1. Baseline: No action (control condition).
2. Naive-30: Pre-registered primary heuristic (30m temporary reassignment, 3m switching delay).
3. Naive-15: Sensitivity analysis heuristic (15m temporary reassignment, 3m switching delay).
4. PRAVAAH: Simulation-based look-ahead optimizer subject to P90 constraints,
   10% improvement margin, 70% win-rate threshold, and information boundary.

Guarantees:
- Seed Separation: Decision seeds (100 reps) and Evaluation seeds (100 reps) are provably disjoint.
- Common Random Numbers (CRN): Evaluation seeds are identical across all policies per replication.
- Information Boundary: Decision simulations observe only announced future events and base arrival profile;
  Scoring simulations observe complete scenario with all events.
- Methodological Symmetry: Naive and PRAVAAH operate on the same legal action space.
- Accurate Waiting Semantics: Waiting person-minutes include accrued wait of citizens queued at the horizon.
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


# =============================================================================
# 1. SEED CONSTANTS & FORMULAS
# =============================================================================

DECISION_BASE = 100_000
EVALUATION_BASE = 200_000
SEED_STEP = 1_000
DEFAULT_BENCHMARK_REPLICATIONS = 100

BENCHMARK_10_SCENARIO_IDS: Tuple[str, ...] = (
    "scenario_a_healthy",
    "scenario_b_pt_pressure",
    "scenario_c_donor_constraint",
    "scenario_d_temporary_intervention",
    "scenario_e_counter_failure",
    "scenario_rto_healthy",
    "scenario_d_announced",
    "scenario_d_unannounced",
    "scenario_e_known",
    "scenario_e_unknown",
)


def get_decision_seeds(
    replications: int = DEFAULT_BENCHMARK_REPLICATIONS,
    base: int = DECISION_BASE,
    step: int = SEED_STEP,
) -> List[int]:
    """
    decision_seed_r = DECISION_BASE + r * SEED_STEP
    For r in 0..N-1. E.g. [100000, 101000, ..., 199000] for N=100.
    """
    return [base + (r * step) for r in range(replications)]


def get_evaluation_seeds(
    replications: int = DEFAULT_BENCHMARK_REPLICATIONS,
    base: int = EVALUATION_BASE,
    step: int = SEED_STEP,
) -> List[int]:
    """
    evaluation_seed_r = EVALUATION_BASE + r * SEED_STEP
    For r in 0..N-1. E.g. [200000, 201000, ..., 299000] for N=100.
    """
    return [base + (r * step) for r in range(replications)]


def get_t_critical_95(df: int) -> float:
    """Returns two-tailed 95% critical value for Student's t distribution with df degrees of freedom."""
    if df <= 0:
        return 1.960
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        14: 2.145, 15: 2.131, 20: 2.086, 29: 2.045, 30: 2.042,
        40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 99: 1.9842,
        100: 1.984, 120: 1.980, 200: 1.972,
    }
    if df in t_table:
        return t_table[df]
    if df >= 30:
        return 1.960 + (2.378 / df)
    sorted_dfs = sorted(t_table.keys())
    for i in range(len(sorted_dfs) - 1):
        if sorted_dfs[i] <= df <= sorted_dfs[i + 1]:
            d1, d2 = sorted_dfs[i], sorted_dfs[i + 1]
            t1, t2 = t_table[d1], t_table[d2]
            return t1 + (t2 - t1) * (df - d1) / (d2 - d1)
    return 1.960


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

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
    standard_error: float
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
            "standard_error": round(self.standard_error, 4),
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
class PairedComparison:
    """Statistical summary of paired differences across replications under Common Random Numbers."""
    comparison_name: str
    mean_difference: float
    standard_error: float
    ci_95_lower: float
    ci_95_upper: float
    win_count: int
    loss_count: int
    tie_count: int
    win_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_name": self.comparison_name,
            "mean_difference": round(self.mean_difference, 4),
            "standard_error": round(self.standard_error, 4),
            "ci_95": [round(self.ci_95_lower, 4), round(self.ci_95_upper, 4)],
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "win_rate": round(self.win_rate, 4),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete benchmarking output for a scenario comparing Baseline, Naive-30, Naive-15, and PRAVAAH."""
    scenario_id: str
    scenario_name: str
    office_code: str
    horizon_minutes: float
    decision_replications: int
    evaluation_replications: int
    baseline: PolicyBenchmarkMetrics
    naive_30: PolicyBenchmarkMetrics
    naive_15: PolicyBenchmarkMetrics
    pravaah: PolicyBenchmarkMetrics
    selected_policy: PolicyType
    selected_candidate_id: Optional[str]
    selected_action_description: str
    pravaah_decision_info: Dict[str, Any]
    vs_baseline: PairedComparison
    vs_naive_30: PairedComparison
    vs_naive_15: PairedComparison
    naive_30_vs_baseline: PairedComparison
    pravaah_beats_baseline: bool
    pravaah_beats_naive_30: bool
    pravaah_beats_naive_15: bool
    naive_15_beats_naive_30: bool
    observations: List[str] = field(default_factory=list)

    @property
    def naive(self) -> PolicyBenchmarkMetrics:
        """Alias to primary Naive-30 policy for backward compatibility."""
        return self.naive_30

    @property
    def replications(self) -> int:
        """Alias to evaluation_replications for backward compatibility."""
        return self.evaluation_replications

    @property
    def improvement_vs_baseline_person_min(self) -> float:
        return self.vs_baseline.mean_difference

    @property
    def pct_improvement_vs_baseline(self) -> float:
        base_w = self.baseline.mean_total_waiting_person_minutes
        return (self.vs_baseline.mean_difference / base_w * 100.0) if base_w > 1e-6 else 0.0

    @property
    def improvement_vs_naive_person_min(self) -> float:
        return self.vs_naive_30.mean_difference

    @property
    def pct_improvement_vs_naive(self) -> float:
        naive_w = self.naive_30.mean_total_waiting_person_minutes
        return (self.vs_naive_30.mean_difference / naive_w * 100.0) if naive_w > 1e-6 else 0.0

    @property
    def pravaah_beats_naive(self) -> bool:
        return self.pravaah_beats_naive_30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "office_code": self.office_code,
            "horizon_minutes": self.horizon_minutes,
            "decision_replications": self.decision_replications,
            "evaluation_replications": self.evaluation_replications,
            "baseline": self.baseline.to_dict(),
            "naive": self.naive_30.to_dict(),
            "naive_30": self.naive_30.to_dict(),
            "naive_15": self.naive_15.to_dict(),
            "pravaah": self.pravaah.to_dict(),
            "selected_policy": self.selected_policy.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_action_description": self.selected_action_description,
            "pravaah_decision_info": self.pravaah_decision_info,
            "vs_baseline": self.vs_baseline.to_dict(),
            "vs_naive_30": self.vs_naive_30.to_dict(),
            "vs_naive_15": self.vs_naive_15.to_dict(),
            "naive_30_vs_baseline": self.naive_30_vs_baseline.to_dict(),
            "improvement_vs_baseline_person_min": round(self.improvement_vs_baseline_person_min, 4),
            "pct_improvement_vs_baseline": round(self.pct_improvement_vs_baseline, 2),
            "improvement_vs_naive_person_min": round(self.improvement_vs_naive_person_min, 4),
            "pct_improvement_vs_naive": round(self.pct_improvement_vs_naive, 2),
            "pravaah_beats_baseline": self.pravaah_beats_baseline,
            "pravaah_beats_naive": self.pravaah_beats_naive_30,
            "pravaah_beats_naive_30": self.pravaah_beats_naive_30,
            "pravaah_beats_naive_15": self.pravaah_beats_naive_15,
            "naive_15_beats_naive_30": self.naive_15_beats_naive_30,
            "observations": self.observations,
        }


# =============================================================================
# 3. HELPER COMPUTATIONS
# =============================================================================

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
    se_wait = std_wait / math.sqrt(rep_count)
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

    # Aggregate individual waits to compute overall P90
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
        standard_error=se_wait,
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


def _compute_paired_comparison(
    control_waits: Sequence[float],
    treatment_waits: Sequence[float],
    comparison_name: str,
) -> PairedComparison:
    """
    Computes paired difference statistics: diff_i = control_i - treatment_i.
    Positive value indicates treatment saved waiting person-minutes.
    """
    n = len(control_waits)
    if n == 0 or len(treatment_waits) != n:
        raise ValueError("control and treatment wait sequences must be non-empty and equal length")

    diffs = [c - t for c, t in zip(control_waits, treatment_waits)]
    mean_diff = sum(diffs) / n

    if n > 1:
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = math.sqrt(var_diff)
        se_diff = std_diff / math.sqrt(n)
        t_crit = get_t_critical_95(n - 1)
        ci_lower = mean_diff - (t_crit * se_diff)
        ci_upper = mean_diff + (t_crit * se_diff)
    else:
        se_diff = 0.0
        ci_lower = mean_diff
        ci_upper = mean_diff

    wins = sum(1 for d in diffs if d > 1e-4)
    losses = sum(1 for d in diffs if d < -1e-4)
    ties = sum(1 for d in diffs if abs(d) <= 1e-4)
    win_rate = wins / n

    return PairedComparison(
        comparison_name=comparison_name,
        mean_difference=mean_diff,
        standard_error=se_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        win_count=wins,
        loss_count=losses,
        tie_count=ties,
        win_rate=win_rate,
    )


def _simulate_scoring_policy(
    scen_config: ScenarioConfig,
    state: SimulationState,
    candidate: Optional[InterventionCandidate],
    seeds: Sequence[int],
) -> List[SimulationResult]:
    """
    Executes scoring simulation replications under the SCORING WORLD.
    Uses complete scenario definition, including all scheduled availability events and bursts.
    """
    engine_interventions = candidate_to_engine_interventions(candidate)
    results: List[SimulationResult] = []

    for seed in seeds:
        rep_rng = random.Random(seed)
        arr_proc = scen_config.build_arrival_process(state, rng=rep_rng, for_decision=False)
        svc_proc = scen_config.build_service_process(rng=rep_rng)

        engine = DiscreteEventEngine(
            horizon_minutes=scen_config.horizon_minutes,
            seed=seed,
            arrival_process=arr_proc,
            service_process=svc_proc,
            interventions=engine_interventions,
            availability_events=list(scen_config.availability_events) if scen_config.availability_events else None,
        )
        res = engine.run(state)
        results.append(res)

    return results


# =============================================================================
# 4. BENCHMARK EXECUTION
# =============================================================================

def run_scenario_benchmark(
    scenario: Union[str, ScenarioConfig],
    replications: Optional[int] = None,
    base_seed: Optional[int] = None,
    decision_seeds: Optional[Sequence[int]] = None,
    evaluation_seeds: Optional[Sequence[int]] = None,
) -> BenchmarkResult:
    """
    Executes an authoritative benchmark for a scenario comparing:
    Baseline, Naive-30, Naive-15, and PRAVAAH.

    Methodological Invariants:
    1. PRAVAAH decision phase uses DECISION SEEDS (default 100 reps) and DECISION WORLD (announced only).
    2. Scoring phase uses EVALUATION SEEDS (default 100 reps) and SCORING WORLD (all events).
    3. Decision seeds and Evaluation seeds are provably disjoint.
    4. Evaluation seeds are shared identically across Baseline, Naive-30, Naive-15, and PRAVAAH (CRN).
    """
    if isinstance(scenario, str):
        scen_config = get_scenario(scenario)
    elif isinstance(scenario, ScenarioConfig):
        scen_config = scenario
    else:
        raise TypeError(f"scenario must be str or ScenarioConfig (got {type(scenario)})")

    rep_count = replications if replications is not None else DEFAULT_BENCHMARK_REPLICATIONS

    # 1. Generate disjoint seeds if not explicitly provided
    if decision_seeds is not None:
        dec_seeds = list(decision_seeds)
    elif base_seed is not None:
        dec_seeds = [base_seed + (r * SEED_STEP) for r in range(rep_count)]
    else:
        dec_seeds = get_decision_seeds(replications=rep_count)

    if evaluation_seeds is not None:
        eval_seeds = list(evaluation_seeds)
    elif base_seed is not None:
        eval_seeds = [base_seed + 100_000 + (r * SEED_STEP) for r in range(rep_count)]
    else:
        eval_seeds = get_evaluation_seeds(replications=rep_count)

    # Invariant: Disjoint seeds
    intersection = set(dec_seeds).intersection(set(eval_seeds))
    if intersection:
        raise ValueError(
            f"Methodological violation: Decision seeds and Evaluation seeds overlap on {len(intersection)} values!"
        )

    # 2. Build initial state for decision
    initial_state = scen_config.build_simulation_state()

    # 3. PHASE A: DECISION PHASE (PRAVAAH Action Selection)
    # Decision world sees announced events only
    eval_config = scen_config.build_evaluator_config(for_decision=True)
    eval_config = EvaluatorConfig(
        horizon_minutes=scen_config.horizon_minutes,
        replications=len(dec_seeds),
        base_seed=dec_seeds[0],
        p90_thresholds=eval_config.p90_thresholds,
        arrival_rates_per_hour=eval_config.arrival_rates_per_hour,
        candidate_durations=scen_config.experiment_params.candidate_durations_minutes,
        switching_delay_min=scen_config.experiment_params.switching_delay_minutes,
        availability_events=eval_config.availability_events,
        arrival_process_factory=lambda st, rng: scen_config.build_arrival_process(st, rng=rng, for_decision=True),
        service_process_factory=lambda rng: scen_config.build_service_process(rng=rng),
    )

    evaluator = InterventionEvaluator(eval_config)
    # Run evaluator using exact decision seeds
    # Temporarily monkey-patch or configure seed list for evaluator
    eval_result = evaluator.evaluate(initial_state, config=eval_config)

    pravaah_selected_cand = eval_result.selected_candidate
    pravaah_policy_type = eval_result.selected_policy
    pravaah_desc = (
        pravaah_selected_cand.description
        if pravaah_selected_cand
        else "No counter reassignment (No intervention required)"
    )

    # 4. PHASE B: EVALUATION PHASE (Scoring World with Evaluation Seeds)
    # Common Random Numbers: Exact same eval_seeds across all 4 policies
    baseline_results = _simulate_scoring_policy(scen_config, initial_state, candidate=None, seeds=eval_seeds)
    baseline_metrics = _compute_policy_metrics(
        policy_type=PolicyType.BASELINE,
        candidate=None,
        results=baseline_results,
        services_map=initial_state.services,
    )

    # Naive-30 (Primary heuristic)
    naive_30_cand = build_naive_intervention(initial_state, duration_min=30.0, switching_delay_min=3.0)
    naive_30_results = _simulate_scoring_policy(scen_config, initial_state, candidate=naive_30_cand, seeds=eval_seeds)
    naive_30_metrics = _compute_policy_metrics(
        policy_type=PolicyType.NAIVE,
        candidate=naive_30_cand,
        results=naive_30_results,
        services_map=initial_state.services,
    )

    # Naive-15 (Sensitivity analysis heuristic)
    naive_15_cand = build_naive_intervention(initial_state, duration_min=15.0, switching_delay_min=3.0)
    naive_15_results = _simulate_scoring_policy(scen_config, initial_state, candidate=naive_15_cand, seeds=eval_seeds)
    naive_15_metrics = _compute_policy_metrics(
        policy_type=PolicyType.NAIVE,
        candidate=naive_15_cand,
        results=naive_15_results,
        services_map=initial_state.services,
    )

    # PRAVAAH (Evaluated with scoring events)
    pravaah_results = _simulate_scoring_policy(scen_config, initial_state, candidate=pravaah_selected_cand, seeds=eval_seeds)
    pravaah_metrics = _compute_policy_metrics(
        policy_type=pravaah_policy_type,
        candidate=pravaah_selected_cand,
        results=pravaah_results,
        services_map=initial_state.services,
    )

    # 5. Paired Comparisons (Control - Treatment: positive = treatment saved waiting person-min)
    vs_base = _compute_paired_comparison(
        control_waits=baseline_metrics.replication_total_waits,
        treatment_waits=pravaah_metrics.replication_total_waits,
        comparison_name="PRAVAAH vs Baseline",
    )
    vs_naive_30 = _compute_paired_comparison(
        control_waits=naive_30_metrics.replication_total_waits,
        treatment_waits=pravaah_metrics.replication_total_waits,
        comparison_name="PRAVAAH vs Naive-30",
    )
    vs_naive_15 = _compute_paired_comparison(
        control_waits=naive_15_metrics.replication_total_waits,
        treatment_waits=pravaah_metrics.replication_total_waits,
        comparison_name="PRAVAAH vs Naive-15",
    )
    naive_30_vs_base = _compute_paired_comparison(
        control_waits=baseline_metrics.replication_total_waits,
        treatment_waits=naive_30_metrics.replication_total_waits,
        comparison_name="Naive-30 vs Baseline",
    )

    # 6. Performance Indicators
    pravaah_beats_base = vs_base.mean_difference > 1e-4
    pravaah_beats_naive_30 = vs_naive_30.mean_difference > 1e-4
    pravaah_beats_naive_15 = vs_naive_15.mean_difference > 1e-4
    naive_15_beats_naive_30 = naive_15_metrics.mean_total_waiting_person_minutes < (
        naive_30_metrics.mean_total_waiting_person_minutes - 1e-4
    )

    # 7. Observations
    observations: List[str] = []
    if pravaah_policy_type == PolicyType.BASELINE:
        observations.append("PRAVAAH concluded NO INTERVENTION REQUIRED (Margin/Feasibility threshold default).")
    elif pravaah_beats_naive_30:
        observations.append(
            f"PRAVAAH outperformed Naive-30 by {vs_naive_30.mean_difference:.2f} person-min (95% CI: [{vs_naive_30.ci_95_lower:.2f}, {vs_naive_30.ci_95_upper:.2f}])."
        )
    elif abs(vs_naive_30.mean_difference) <= 1e-4:
        observations.append("PRAVAAH tied Naive-30 in evaluated waiting person-minutes.")
    else:
        observations.append(
            f"Naive-30 achieved lower waiting person-minutes than PRAVAAH by {-vs_naive_30.mean_difference:.2f} person-min."
        )

    if naive_15_beats_naive_30:
        diff_n = naive_30_metrics.mean_total_waiting_person_minutes - naive_15_metrics.mean_total_waiting_person_minutes
        observations.append(
            f"Sensitivity note: Naive-15 outperformed Naive-30 by {diff_n:.2f} person-min."
        )

    decision_info = {
        "selected_candidate_id": pravaah_selected_cand.candidate_id if pravaah_selected_cand else None,
        "selected_policy": pravaah_policy_type.value,
        "decision_replications": len(dec_seeds),
        "decision_mean_wait": round(eval_result.total_waiting_person_minutes, 4),
        "decision_margin": eval_result.explanation.get("decision_margin", {}),
        "is_feasible": eval_result.is_feasible,
        "decision_notes": eval_result.explanation.get("decision_margin", {}).get("decision_notes", []),
    }

    return BenchmarkResult(
        scenario_id=scen_config.scenario_id,
        scenario_name=scen_config.scenario_name,
        office_code=scen_config.office_code,
        horizon_minutes=scen_config.horizon_minutes,
        decision_replications=len(dec_seeds),
        evaluation_replications=len(eval_seeds),
        baseline=baseline_metrics,
        naive_30=naive_30_metrics,
        naive_15=naive_15_metrics,
        pravaah=pravaah_metrics,
        selected_policy=pravaah_policy_type,
        selected_candidate_id=pravaah_selected_cand.candidate_id if pravaah_selected_cand else None,
        selected_action_description=pravaah_desc,
        pravaah_decision_info=decision_info,
        vs_baseline=vs_base,
        vs_naive_30=vs_naive_30,
        vs_naive_15=vs_naive_15,
        naive_30_vs_baseline=naive_30_vs_base,
        pravaah_beats_baseline=pravaah_beats_base,
        pravaah_beats_naive_30=pravaah_beats_naive_30,
        pravaah_beats_naive_15=pravaah_beats_naive_15,
        naive_15_beats_naive_30=naive_15_beats_naive_30,
        observations=observations,
    )


def run_all_benchmarks(
    scenarios: Optional[Sequence[str]] = None,
    replications: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Executes benchmarks across requested scenarios (default: all 10 registered benchmark scenarios).
    """
    target_ids = list(scenarios) if scenarios is not None else list(BENCHMARK_10_SCENARIO_IDS)
    results: Dict[str, BenchmarkResult] = {}

    for scen_id in target_ids:
        res = run_scenario_benchmark(
            scen_id,
            replications=replications,
            base_seed=base_seed,
        )
        results[scen_id] = res

    return results
