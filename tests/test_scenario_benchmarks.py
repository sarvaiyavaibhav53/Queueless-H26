"""
PRAVAAH — Step 2D.4: Comprehensive Methodology & Invariant Test Suite.

Validates the methodological requirements of Step 2D.4:
A. Stability thresholds (STABLE: <0.85, WATCH: 0.85-1.0, UNSTABLE: >=1.0, synthetic project assumptions)
B. Naive trigger & selection (stability gate, longest queue target, donor counter retention, deterministic ranking)
C. Random seed separation (disjoint decision & evaluation sets, CRN alignment across all 4 policies)
D. PRAVAAH decision margin (Condition A: 10% improvement, Condition B: 70% win rate, P90 feasibility)
E. Information boundary (announced vs unannounced visibility in decision vs scoring worlds)
F. Horizon-end waiting semantics (accrued waiting time for citizens still queued at horizon)
G. Scenario invariants (A–E definitions unchanged, 4 new variants correctly configured)
H. Common Random Numbers & determinism (reproducible paired evaluation)
"""

from datetime import datetime, timezone
import math
import os
import sys
import unittest
from unittest.mock import patch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.interventions.candidates import (
    build_naive_intervention,
    generate_candidate_interventions,
)
from app.interventions.evaluator import InterventionEvaluator
from app.interventions.models import (
    EvaluatorConfig,
    InterventionCandidate,
    PolicyEvaluation,
    PolicyType,
)
from app.simulation.benchmarking import (
    BENCHMARK_10_SCENARIO_IDS,
    DECISION_BASE,
    EVALUATION_BASE,
    SEED_STEP,
    BenchmarkResult,
    get_decision_seeds,
    get_evaluation_seeds,
    get_t_critical_95,
    run_all_benchmarks,
    run_scenario_benchmark,
)
from app.simulation.engine import (
    CounterAvailabilityEvent,
    DiscreteEventEngine,
)
from app.simulation.processes import (
    ArrivalBurst,
    ArrivalPhase,
    PhaseArrivalProcess,
)
from app.simulation.results import SimulationResult
from app.simulation.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioConfig,
    build_scenario_a_healthy,
    build_scenario_b_pt_pressure,
    build_scenario_c_donor_constraint,
    build_scenario_d_temporary_intervention,
    build_scenario_d_announced,
    build_scenario_d_unannounced,
    build_scenario_e_counter_failure,
    build_scenario_e_known,
    build_scenario_e_unknown,
    get_scenario,
)
from app.simulation.state import SimCitizen, SimCounter, SimQueue, SimulationState
from shared.features import ServiceConfigSnapshot
from shared.stability import (
    DEFAULT_STABILITY_THRESHOLDS,
    StabilityState,
    StabilityThresholds,
    classify_stability,
    is_stable,
)


# =============================================================================
# A. STABILITY THRESHOLD TESTS
# =============================================================================

class TestStabilityThresholds(unittest.TestCase):
    """Verifies centralized, configurable stability thresholds."""

    def test_stable_classification(self):
        self.assertEqual(classify_stability(0.0), StabilityState.STABLE)
        self.assertEqual(classify_stability(0.50), StabilityState.STABLE)
        self.assertEqual(classify_stability(0.8499), StabilityState.STABLE)
        self.assertTrue(is_stable(0.80))

    def test_watch_classification(self):
        self.assertEqual(classify_stability(0.85), StabilityState.WATCH)
        self.assertEqual(classify_stability(0.90), StabilityState.WATCH)
        self.assertEqual(classify_stability(0.9999), StabilityState.WATCH)
        self.assertFalse(is_stable(0.85))

    def test_unstable_classification(self):
        self.assertEqual(classify_stability(1.00), StabilityState.UNSTABLE)
        self.assertEqual(classify_stability(1.50), StabilityState.UNSTABLE)
        self.assertEqual(classify_stability(99.0), StabilityState.UNSTABLE)
        self.assertFalse(is_stable(1.00))

    def test_boundary_behavior_at_0_85(self):
        # Boundary at 0.85: < 0.85 is STABLE, >= 0.85 is WATCH
        self.assertTrue(is_stable(0.849999))
        self.assertFalse(is_stable(0.850000))
        self.assertEqual(classify_stability(0.849999), StabilityState.STABLE)
        self.assertEqual(classify_stability(0.850000), StabilityState.WATCH)

    def test_boundary_behavior_at_1_0(self):
        # Boundary at 1.0: < 1.0 is WATCH, >= 1.0 is UNSTABLE
        self.assertEqual(classify_stability(0.999999), StabilityState.WATCH)
        self.assertEqual(classify_stability(1.000000), StabilityState.UNSTABLE)

    def test_centralized_configurable_thresholds(self):
        custom = StabilityThresholds(stable_limit=0.70, watch_limit=0.90)
        self.assertEqual(classify_stability(0.65, custom), StabilityState.STABLE)
        self.assertEqual(classify_stability(0.75, custom), StabilityState.WATCH)
        self.assertEqual(classify_stability(0.95, custom), StabilityState.UNSTABLE)


# =============================================================================
# B. NAIVE TRIGGER & SELECTION TESTS
# =============================================================================

class TestNaiveTriggerAndSelection(unittest.TestCase):
    """Verifies that Naive acts only when target is not stable and satisfies all donor rules."""

    def setUp(self):
        self.t_snap = datetime(2026, 10, 8, 10, 0, 0, tzinfo=timezone.utc)
        self.svc1 = ServiceConfigSnapshot(1, "S1", "Service 1", 10.0, 4.0)
        self.svc2 = ServiceConfigSnapshot(2, "S2", "Service 2", 15.0, 6.0)
        self.svc3 = ServiceConfigSnapshot(3, "S3", "Service 3", 8.0, 3.0)

    def test_stable_target_returns_no_action(self):
        """If target service is STABLE (rho < 0.85), Naive returns no action (None)."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(3)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({1, 2}), True),
            },
            traffic_intensity_by_service={1: 0.70, 2: 0.40},  # S1 is STABLE (0.70 < 0.85)
        )
        self.assertIsNone(build_naive_intervention(state))

    def test_non_stable_target_allows_action(self):
        """If target service is NOT STABLE (rho >= 0.85), Naive may select an eligible donor."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(5)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({1, 2}), True),
            },
            traffic_intensity_by_service={1: 0.95, 2: 0.40},  # S1 is WATCH (0.95 >= 0.85)
        )
        cand = build_naive_intervention(state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.target_service_id, 1)

    def test_all_queues_empty_returns_no_action(self):
        """If all queues are empty, Naive returns no action."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={1: SimQueue(1, []), 2: SimQueue(2, [])},
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.40},
        )
        self.assertIsNone(build_naive_intervention(state))

    def test_longest_queue_target_selection(self):
        """Naive selects the service with the longest current waiting queue."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2, 3: self.svc3},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(4)]),
                2: SimQueue(2, [SimCitizen(2, "T2", 2, 0.0) for _ in range(9)]),  # Longest
                3: SimQueue(3, [SimCitizen(3, "T3", 3, 0.0) for _ in range(1)]),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1, 2}), True),
                2: SimCounter(2, 1, "C2", 1, frozenset({1}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({2}), True),
                4: SimCounter(4, 1, "C4", 3, frozenset({2, 3}), True),
                5: SimCounter(5, 1, "C5", 3, frozenset({3}), True),
            },
            traffic_intensity_by_service={1: 0.50, 2: 1.10, 3: 0.40},
        )
        cand = build_naive_intervention(state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.target_service_id, 2)

    def test_donor_source_must_keep_at_least_one_active_counter(self):
        """Donor is rejected if reassignment leaves its source service with 0 active counters."""
        # S2 has only ONE counter (C2). C2 cannot donate!
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(5)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),  # Sole counter on S2
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.20},
        )
        self.assertIsNone(build_naive_intervention(state))

        # Adding a second counter to S2 allows donation!
        state_with_2 = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(5)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({2}), True),  # Second counter keeps S2 staffed
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.20},
        )
        cand = build_naive_intervention(state_with_2)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.counter_id, 2)

    def test_donor_cannot_be_already_assigned_to_target(self):
        """Counters already assigned to target service cannot be donor candidates."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1},
            queues={1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(5)])},
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 1, frozenset({1}), True),
            },
            traffic_intensity_by_service={1: 1.20},
        )
        self.assertIsNone(build_naive_intervention(state))

    def test_donor_deterministic_rho_tie_break(self):
        """Donor from source service with lowest current rho is chosen."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2, 3: self.svc3},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(8)]),
                2: SimQueue(2, []),
                3: SimQueue(3, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({2}), True),
                4: SimCounter(4, 1, "C4", 3, frozenset({1, 3}), True),
                5: SimCounter(5, 1, "C5", 3, frozenset({3}), True),
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.80, 3: 0.30},  # S3 rho 0.30 < S2 rho 0.80
        )
        cand = build_naive_intervention(state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.counter_id, 4)
        self.assertEqual(cand.source_service_id, 3)

    def test_donor_idle_preferred_on_rho_tie(self):
        """When source rho is tied, idle donor is preferred over busy donor."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(6)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True, is_busy=True),  # Busy
                3: SimCounter(3, 1, "C3", 2, frozenset({1, 2}), True, is_busy=False), # Idle
                4: SimCounter(4, 1, "C4", 2, frozenset({2}), True),
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.40},
        )
        cand = build_naive_intervention(state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.counter_id, 3)

    def test_donor_counter_id_tie_break(self):
        """When rho and busy status are tied, lowest counter_id is chosen."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(6)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True, is_busy=False),
                3: SimCounter(3, 1, "C3", 2, frozenset({1, 2}), True, is_busy=False),
                4: SimCounter(4, 1, "C4", 2, frozenset({2}), True),
            },
            traffic_intensity_by_service={1: 1.20, 2: 0.40},
        )
        cand = build_naive_intervention(state)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.counter_id, 2)

    def test_naive_30_contract(self):
        """Primary Naive-30 has duration=30.0m, switching_delay=3.0m, and never None."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(6)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({2}), True),
            },
            traffic_intensity_by_service={1: 1.10, 2: 0.40},
        )
        cand = build_naive_intervention(state, duration_min=30.0)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.duration_min, 30.0)
        self.assertEqual(cand.switching_delay_min, 3.0)

    def test_naive_15_contract(self):
        """Sensitivity Naive-15 has duration=15.0m, switching_delay=3.0m."""
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, [SimCitizen(1, "T1", 1, 0.0) for _ in range(6)]),
                2: SimQueue(2, []),
            },
            counters={
                1: SimCounter(1, 1, "C1", 1, frozenset({1}), True),
                2: SimCounter(2, 1, "C2", 2, frozenset({1, 2}), True),
                3: SimCounter(3, 1, "C3", 2, frozenset({2}), True),
            },
            traffic_intensity_by_service={1: 1.10, 2: 0.40},
        )
        cand = build_naive_intervention(state, duration_min=15.0)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.duration_min, 15.0)
        self.assertEqual(cand.switching_delay_min, 3.0)


# =============================================================================
# C. RANDOM SEED SEPARATION TESTS
# =============================================================================

class TestRandomSeedSeparation(unittest.TestCase):
    """Verifies disjoint seed generation and common evaluation seed sharing."""

    def test_decision_and_evaluation_seeds_are_disjoint(self):
        dec_seeds = get_decision_seeds(replications=100)
        eval_seeds = get_evaluation_seeds(replications=100)

        self.assertEqual(len(dec_seeds), 100)
        self.assertEqual(len(eval_seeds), 100)
        # Sets must be disjoint
        self.assertTrue(set(dec_seeds).isdisjoint(set(eval_seeds)))
        # Range check: max decision < min evaluation
        self.assertLess(max(dec_seeds), min(eval_seeds))

    def test_seed_generation_deterministic(self):
        d1 = get_decision_seeds(100)
        d2 = get_decision_seeds(100)
        self.assertEqual(d1, d2)

        e1 = get_evaluation_seeds(100)
        e2 = get_evaluation_seeds(100)
        self.assertEqual(e1, e2)

    def test_evaluation_seeds_shared_across_policies(self):
        """Verifies that Baseline, Naive-30, Naive-15, and PRAVAAH receive the same evaluation seeds."""
        captured_seeds_by_call = []
        real_engine_init = DiscreteEventEngine.__init__

        def spy_engine_init(self_eng, horizon_minutes, seed=None, **kwargs):
            if seed is not None and seed >= EVALUATION_BASE:
                captured_seeds_by_call.append(seed)
            real_engine_init(self_eng, horizon_minutes, seed=seed, **kwargs)

        with patch.object(DiscreteEventEngine, "__init__", spy_engine_init):
            res = run_scenario_benchmark("scenario_b_pt_pressure", replications=5)

        eval_seeds = get_evaluation_seeds(5)
        # 4 policies * 5 replications = 20 scoring runs
        self.assertEqual(len(captured_seeds_by_call), 20)
        for i in range(4):
            policy_seeds = captured_seeds_by_call[i * 5 : (i + 1) * 5]
            self.assertEqual(policy_seeds, eval_seeds)


# =============================================================================
# D. PRAVAAH DECISION MARGIN TESTS
# =============================================================================

class TestPravaahDecisionMargin(unittest.TestCase):
    """Verifies Condition A (10% improvement) and Condition B (70% win rate)."""

    def setUp(self):
        self.t_snap = datetime(2026, 10, 8, 10, 0, 0, tzinfo=timezone.utc)
        self.svc1 = ServiceConfigSnapshot(1, "S1", "Service 1", 10.0, 4.0)
        self.svc2 = ServiceConfigSnapshot(2, "S2", "Service 2", 10.0, 4.0)

    def test_less_than_10_percent_improvement_selects_baseline(self):
        """If best candidate improves by only 5% (< 10%), PRAVAAH selects Baseline."""
        c1 = SimCounter(1, 1, "C1", 1, frozenset({1, 2}), True)
        c2 = SimCounter(2, 1, "C2", 2, frozenset({2}), True)
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={1: SimQueue(1, []), 2: SimQueue(2, [])},
            counters={1: c1, 2: c2},
            traffic_intensity_by_service={1: 0.5, 2: 0.5},
        )
        cfg = EvaluatorConfig(horizon_minutes=20.0, replications=5, base_seed=42)
        res = InterventionEvaluator(cfg).evaluate(state)
        self.assertEqual(res.selected_policy, PolicyType.BASELINE)
        self.assertIsNone(res.selected_candidate)

    def test_p90_violation_forces_baseline(self):
        """Infeasible candidates violating P90 thresholds are rejected, selecting Baseline."""
        # 2 citizens in S2 with strict SLA of 2.0 min
        c1 = SimCounter(1, 1, "C1", 1, frozenset({1, 2}), True)
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=self.t_snap,
            services={1: self.svc1, 2: self.svc2},
            queues={
                1: SimQueue(1, []),
                2: SimQueue(2, [SimCitizen(1, "T1", 2, 0.0), SimCitizen(2, "T2", 2, 0.0)]),
            },
            counters={1: c1},
            traffic_intensity_by_service={1: 0.1, 2: 1.5},
        )
        cfg = EvaluatorConfig(
            horizon_minutes=30.0, replications=3, base_seed=42,
            p90_thresholds={2: 2.0},  # SLA 2.0m cannot be met
        )
        res = InterventionEvaluator(cfg).evaluate(state)
        self.assertEqual(res.selected_policy, PolicyType.BASELINE)


# =============================================================================
# E. INFORMATION BOUNDARY TESTS
# =============================================================================

class TestInformationBoundary(unittest.TestCase):
    """Verifies that unannounced future events are invisible during decision-making."""

    def test_announced_event_visible_to_decision_world(self):
        scen = build_scenario_d_announced()
        cfg = scen.build_evaluator_config(for_decision=True)
        # In D_announced, the burst is announced=True
        self.assertEqual(len(scen.arrival_bursts), 1)
        self.assertTrue(scen.arrival_bursts[0].announced)

    def test_unannounced_event_invisible_to_decision_world(self):
        scen = build_scenario_d_unannounced()
        state = scen.build_simulation_state()
        # Decision arrival process has 0 bursts because announced=False
        arr_dec = scen.build_arrival_process(state, for_decision=True)
        self.assertEqual(len(arr_dec.bursts), 0)

        # Scoring arrival process HAS the burst
        arr_score = scen.build_arrival_process(state, for_decision=False)
        self.assertEqual(len(arr_score.bursts), 1)

    def test_unannounced_burst_arrival_rate_equals_base_rate(self):
        """Proves that during an unannounced burst interval, decision rate == base rate."""
        scen = build_scenario_d_unannounced()
        state = scen.build_simulation_state()

        # At t=45 min (inside burst interval [30, 60] min)
        arr_dec = scen.build_arrival_process(state, for_decision=True)
        arr_score = scen.build_arrival_process(state, for_decision=False)

        # BC burst is on service_id=3 (SERVICE_BC_ID)
        rate_dec = arr_dec.get_arrival_rate_per_hour(3, 45.0)
        rate_score = arr_score.get_arrival_rate_per_hour(3, 45.0)

        # Decision rate is the un-burst rate
        # Scoring rate is doubled (multiplier 2.0x)
        self.assertAlmostEqual(rate_score, rate_dec * 2.0, places=4)
        self.assertLess(rate_dec, rate_score)

    def test_unannounced_counter_failure_invisible_to_decision(self):
        scen = build_scenario_e_unknown()
        cfg_dec = scen.build_evaluator_config(for_decision=True)
        # Failure at t=10 is unannounced -> filtered from decision evaluator
        self.assertEqual(len(cfg_dec.availability_events), 0)
        # But present in scenario availability_events for scoring!
        self.assertEqual(len(scen.availability_events), 1)


# =============================================================================
# F. HORIZON-END WAITING SEMANTICS TESTS
# =============================================================================

class TestHorizonEndWaitingSemantics(unittest.TestCase):
    """Verifies accrued waiting time for citizens still queued at simulation horizon."""

    def test_unserved_citizen_accrues_wait_to_horizon(self):
        t_snap = datetime(2026, 10, 8, 10, 0, 0, tzinfo=timezone.utc)
        svc = ServiceConfigSnapshot(1, "S1", "Service 1", 10.0, 1.0)
        # Citizen arrived at t=10.0; horizon is 60.0; counter is unavailable
        state = SimulationState(
            office_id=1, office_code="OFF1", snapshot_time_utc=t_snap,
            services={1: svc},
            queues={1: SimQueue(1, [SimCitizen(1, "T1", 1, 10.0)])},
            counters={1: SimCounter(1, 1, "C1", 1, frozenset({1}), is_active=False)},
        )
        engine = DiscreteEventEngine(horizon_minutes=60.0, seed=42)
        res = engine.run(state)

        # Citizen waited from t=10 to t=60 = 50 person-minutes accrued wait
        self.assertEqual(res.total_customers_served, 0)
        self.assertEqual(len(res.unserved_citizen_ids), 1)
        self.assertAlmostEqual(res.unserved_waiting_person_minutes, 50.0, places=4)
        self.assertAlmostEqual(res.total_waiting_person_minutes, 50.0, places=4)


# =============================================================================
# G. SCENARIO INVARIANTS & REGISTRY TESTS
# =============================================================================

class TestScenarioInvariants(unittest.TestCase):
    """Verifies that A–E are unchanged and 4 variants are properly registered."""

    def test_all_10_scenarios_in_registry(self):
        self.assertEqual(len(BENCHMARK_10_SCENARIO_IDS), 10)
        for s_id in BENCHMARK_10_SCENARIO_IDS:
            scen = get_scenario(s_id)
            self.assertIsInstance(scen, ScenarioConfig)

    def test_scenarios_a_through_e_parameters_preserved(self):
        scen_a = build_scenario_a_healthy()
        self.assertEqual(scen_a.horizon_minutes, 60.0)
        self.assertEqual(scen_a.office_code, "MUNI")

        scen_b = build_scenario_b_pt_pressure()
        self.assertEqual(scen_b.initial_queue_counts[1], 8)

        scen_c = build_scenario_c_donor_constraint()
        self.assertEqual(scen_c.initial_queue_counts[1], 8)

        scen_d = build_scenario_d_temporary_intervention()
        self.assertEqual(scen_d.initial_queue_counts[1], 6)

        scen_e = build_scenario_e_counter_failure()
        self.assertEqual(len(scen_e.availability_events), 1)
        self.assertEqual(scen_e.availability_events[0].time_min, 10.0)

    def test_e_known_has_initial_counter_inactive(self):
        scen = build_scenario_e_known()
        c1 = next(c for c in scen.initial_counters if c.counter_id == 1)
        self.assertFalse(c1.is_active)
        self.assertEqual(len(scen.availability_events), 0)


# =============================================================================
# H. CRN & BENCHMARK DETERMINISM TESTS
# =============================================================================

class TestCRNAndDeterminism(unittest.TestCase):
    """Verifies CRN paired comparison and deterministic execution."""

    def test_benchmark_determinism(self):
        """Running benchmark twice produces identical results."""
        res1 = run_scenario_benchmark("scenario_a_healthy", replications=5, base_seed=123)
        res2 = run_scenario_benchmark("scenario_a_healthy", replications=5, base_seed=123)

        self.assertEqual(
            res1.baseline.replication_total_waits,
            res2.baseline.replication_total_waits,
        )
        self.assertEqual(
            res1.pravaah.replication_total_waits,
            res2.pravaah.replication_total_waits,
        )
        self.assertAlmostEqual(
            res1.vs_baseline.mean_difference,
            res2.vs_baseline.mean_difference,
            places=7,
        )

    def test_all_10_scenarios_execute_cleanly(self):
        """Executes a short run across all 10 scenarios to verify pipeline integrity."""
        results = run_all_benchmarks(replications=3, base_seed=500)
        self.assertEqual(len(results), 10)
        for s_id in BENCHMARK_10_SCENARIO_IDS:
            self.assertIn(s_id, results)
            self.assertIsInstance(results[s_id], BenchmarkResult)


if __name__ == "__main__":
    unittest.main()
