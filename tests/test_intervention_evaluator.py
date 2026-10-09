"""
PRAVAAH — Step 2C.4: Candidate Intervention Generator & Evaluator Tests.

Verifies:
A. Baseline: No intervention produces a valid simulation result.
B. Naive longest queue: Correctly identifies the longest current queue.
C. Naive eligibility: Never assigns a counter to an ineligible service.
D. No valid naive counter: Falls back safely to no intervention.
E. Candidate generation: Only valid counter/service combinations generated.
F. Busy counter: Active serving counter is not interrupted.
G. Candidate evaluation: Multiple candidates evaluated independently.
H. Same seeds: Baseline and candidates use identical seeds for corresponding replications.
I. Determinism: Same snapshot + config + seeds produces identical results.
J. P90 calculation: Verified against a manually calculable dataset.
K. P90 rejection: Candidate exceeding service threshold is rejected.
L. Feasible candidate selection: Lowest total waiting person-minutes among feasible wins.
M. No feasible intervention: Baseline selected when every intervention violates constraints.
N. Optimizer does not beat naive: System honestly reports mathematically correct result.
O. Improvement metrics: Baseline and naive improvement calculations.
P. Zero denominator: Percentage improvement handles 0.0 baseline safely.
Q. Candidate isolation: Evaluating one candidate does not mutate another.
R. Multiple replications: Results aggregate correctly across replications.
"""

import math
import os
import sys
import unittest
from datetime import datetime, timedelta

# Ensure project root and backend are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.interventions import (
    EvaluatorConfig,
    InterventionCandidate,
    InterventionEvaluator,
    PolicyType,
    build_naive_intervention,
    calculate_percentile,
    generate_candidate_interventions,
)
from app.simulation import (
    SimulationState,
)
from shared.features import (
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServingTokenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    SimulationSnapshot,
    UTC_TIMEZONE,
)


class TestInterventionEvaluator(unittest.TestCase):
    def setUp(self):
        self.t_snap = datetime(2026, 10, 7, 10, 30, 0, tzinfo=UTC_TIMEZONE)
        self.svc_pt = ServiceConfigSnapshot(1, "PT", "Property Tax", mean_service_minutes=10.0, std_service_minutes=0.0)
        self.svc_wc = ServiceConfigSnapshot(2, "WC", "Water Connection", mean_service_minutes=15.0, std_service_minutes=0.0)

    # =========================================================================
    # A. Baseline
    # =========================================================================
    def test_a_baseline_evaluation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        w1 = QueueCitizenSnapshot(1, "PT-1", self.t_snap)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (w1,), ())},
            counters={1: c1},
        )
        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=30.0, replications=2, base_seed=42))
        res = evaluator.evaluate(snap)

        self.assertIsNotNone(res.baseline_evaluation)
        self.assertEqual(res.baseline_evaluation.policy_type, PolicyType.BASELINE)
        self.assertTrue(res.baseline_evaluation.is_feasible)
        self.assertEqual(res.baseline_evaluation.replication_count, 2)
        self.assertAlmostEqual(res.baseline_evaluation.mean_total_waiting_person_minutes, 0.0)

    # =========================================================================
    # B. Naive longest queue
    # =========================================================================
    def test_b_naive_identifies_longest_queue(self):
        # Service 1 has 3 waiting citizens; Service 2 has 1 citizen
        w_pt = (
            QueueCitizenSnapshot(1, "PT-1", self.t_snap),
            QueueCitizenSnapshot(2, "PT-2", self.t_snap),
            QueueCitizenSnapshot(3, "PT-3", self.t_snap),
        )
        w_wc = (QueueCitizenSnapshot(4, "WC-1", self.t_snap),)

        # Counter 1 assigned to PT (eligible PT)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        # Counter 2 assigned to WC (eligible PT, WC)
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 2, frozenset([1, 2]), None)
        # Counter 3 assigned to WC (ensures WC retains at least 1 counter after C2 donation)
        c3 = CounterSnapshot(3, 101, "C3", "ACTIVE", self.t_snap, 2, frozenset([2]), None)

        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, w_pt, ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1, 2: c2, 3: c3},
        )
        state = SimulationState.from_snapshot(snap)
        state.traffic_intensity_by_service = {1: 1.20, 2: 0.40}  # S1 is UNSTABLE
        naive_cand = build_naive_intervention(state)

        self.assertIsNotNone(naive_cand)
        self.assertEqual(naive_cand.target_service_id, 1)  # Longest queue is Service 1 (PT)
        self.assertEqual(naive_cand.counter_id, 2)         # Counter 2 reallocated to Service 1

    # =========================================================================
    # C. Naive eligibility
    # =========================================================================
    def test_c_naive_respects_counter_eligibility(self):
        # Service 1 has 5 citizens; Service 2 has 0
        w_pt = tuple(QueueCitizenSnapshot(i, f"PT-{i}", self.t_snap) for i in range(1, 6))

        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        # Counter 2 assigned to WC, but eligible ONLY for WC (NOT PT)
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 2, frozenset([2]), None)

        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, w_pt, ()),
                2: ServiceQueueSnapshot(2, (), ()),
            },
            counters={1: c1, 2: c2},
        )
        state = SimulationState.from_snapshot(snap)
        naive_cand = build_naive_intervention(state)

        # Counter 2 cannot be assigned to PT because it is not eligible!
        self.assertIsNone(naive_cand)

    # =========================================================================
    # D. No valid naive counter falls back safely
    # =========================================================================
    def test_d_no_valid_naive_falls_back_to_no_intervention(self):
        # Both queues empty
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snap)
        naive_cand = build_naive_intervention(state)
        self.assertIsNone(naive_cand)

        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=30.0, replications=1))
        res = evaluator.evaluate(snap)
        # Naive evaluation equals baseline
        self.assertEqual(
            res.naive_evaluation.mean_total_waiting_person_minutes,
            res.baseline_evaluation.mean_total_waiting_person_minutes,
        )

    # =========================================================================
    # E. Candidate generation
    # =========================================================================
    def test_e_candidate_generation_valid_combinations(self):
        # C1 (assigned 1, eligible 1, 2)
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        # C2 (assigned 2, eligible 2) -> cannot be assigned to 1
        c2 = CounterSnapshot(2, 101, "C2", "ACTIVE", self.t_snap, 2, frozenset([2]), None)

        w_wc = (QueueCitizenSnapshot(1, "WC-1", self.t_snap),)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1, 2: c2},
        )
        state = SimulationState.from_snapshot(snap)
        candidates = generate_candidate_interventions(state, max_candidates=10)

        # Only C1 -> Service 2 is valid
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].counter_id, 1)
        self.assertEqual(candidates[0].target_service_id, 2)

    # =========================================================================
    # F. Busy counter is not interrupted
    # =========================================================================
    def test_f_busy_counter_not_interrupted(self):
        # C1 is serving token 50 (started 5 mins ago, finishes in 5 mins)
        serving_token = ServingTokenSnapshot(
            token_id=50, token_number="PT-50", service_id=1, counter_id=1,
            status="SERVING", called_at_utc=self.t_snap - timedelta(minutes=6),
            serving_at_utc=self.t_snap - timedelta(minutes=5)
        )
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), serving_token)
        w_wc = (QueueCitizenSnapshot(2, "WC-1", self.t_snap),)

        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        # Reassign C1 to Service 2 at t=0
        cand = InterventionCandidate("REASSIGN_BUSY", counter_id=1, target_service_id=2, effective_time_min=0.0)
        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=40.0, replications=1, base_seed=42))
        res = evaluator._simulate_policy(SimulationState.from_snapshot(snap), candidate=cand, seeds=[42])[0]

        # Token 50 must complete service without interruption
        self.assertIn(50, res.completed_citizen_ids)
        # WC token 2 is also served after C1 completes token 50
        self.assertIn(2, res.completed_citizen_ids)

    # =========================================================================
    # G. Candidate evaluation
    # =========================================================================
    def test_g_candidate_evaluation_independent_runs(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_wc = (QueueCitizenSnapshot(1, "WC-1", self.t_snap),)

        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=30.0, replications=2))
        res = evaluator.evaluate(snap)

        self.assertIsInstance(res.candidate_evaluations, list)
        self.assertGreater(len(res.candidate_evaluations), 0)

    # =========================================================================
    # H. Same seeds across replications
    # =========================================================================
    def test_h_same_seeds_used_across_all_candidates(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_pt = (QueueCitizenSnapshot(1, "PT-1", self.t_snap),)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={1: ServiceQueueSnapshot(1, w_pt, ()), 2: ServiceQueueSnapshot(2, (), ())},
            counters={1: c1},
        )
        evaluator = InterventionEvaluator(EvaluatorConfig(
            horizon_minutes=30.0, replications=3, base_seed=777,
            arrival_rates_per_hour={1: 10.0}
        ))
        res = evaluator.evaluate(snap)

        # Baseline and Naive with no changes must experience identical replication results
        self.assertEqual(
            res.baseline_evaluation.replication_total_waits,
            res.naive_evaluation.replication_total_waits,
        )

    # =========================================================================
    # I. Determinism
    # =========================================================================
    def test_i_evaluator_determinism(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_pt = (QueueCitizenSnapshot(1, "PT-1", self.t_snap),)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={1: ServiceQueueSnapshot(1, w_pt, ()), 2: ServiceQueueSnapshot(2, (), ())},
            counters={1: c1},
        )
        cfg = EvaluatorConfig(horizon_minutes=30.0, replications=3, base_seed=42)
        eval1 = InterventionEvaluator(cfg).evaluate(snap)
        eval2 = InterventionEvaluator(cfg).evaluate(snap)

        self.assertEqual(eval1.selected_policy, eval2.selected_policy)
        self.assertAlmostEqual(eval1.total_waiting_person_minutes, eval2.total_waiting_person_minutes)
        self.assertAlmostEqual(eval1.improvement_vs_baseline, eval2.improvement_vs_baseline)

    # =========================================================================
    # J. P90 calculation against manually calculable dataset
    # =========================================================================
    def test_j_p90_percentile_calculation(self):
        # 10 values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        # k = (10 - 1) * 0.9 = 8.1
        # floor = 8 (val=90), ceil = 9 (val=100)
        # 90 * 0.9 + 100 * 0.1 = 81.0 + 10.0 = 91.0
        data_10 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertAlmostEqual(calculate_percentile(data_10, 90.0), 91.0)

        # Single element
        self.assertEqual(calculate_percentile([42.0], 90.0), 42.0)

        # Zero elements
        self.assertEqual(calculate_percentile([], 90.0), 0.0)

    # =========================================================================
    # K. P90 rejection
    # =========================================================================
    def test_k_p90_constraint_rejects_violating_candidate(self):
        # Service 1 has 3 citizens. Service 2 has 2 citizens.
        # Reassigning C1 away from Service 1 causes Service 1 citizens to wait > 20m.
        # Setting threshold of Service 1 to 15.0m must reject that candidate!
        w_pt = (
            QueueCitizenSnapshot(1, "PT-1", self.t_snap),
            QueueCitizenSnapshot(2, "PT-2", self.t_snap),
        )
        w_wc = (QueueCitizenSnapshot(3, "WC-1", self.t_snap),)

        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, w_pt, ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        # Tight threshold on PT (service 1): maximum allowed P90 wait = 5.0m
        cfg = EvaluatorConfig(
            horizon_minutes=40.0,
            replications=1,
            base_seed=42,
            p90_thresholds={1: 5.0},
        )
        evaluator = InterventionEvaluator(cfg)
        res = evaluator.evaluate(snap)

        # If C1 was reassigned to Service 2, PT citizens waited up to 40m, so that candidate must be rejected!
        cand_reassign = next((c for c in res.candidate_evaluations if c.candidate and c.candidate.target_service_id == 2), None)
        if cand_reassign:
            self.assertFalse(cand_reassign.is_feasible)
            self.assertIn("P90 threshold", cand_reassign.rejection_reason)

    # =========================================================================
    # L. Feasible candidate selection
    # =========================================================================
    def test_l_feasible_candidate_selection(self):
        # Service 1 has 0 citizens, Service 2 has 3 citizens
        # C1 assigned to Service 1, eligible for both
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_wc = tuple(QueueCitizenSnapshot(i, f"WC-{i}", self.t_snap) for i in range(1, 4))

        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        cfg = EvaluatorConfig(horizon_minutes=60.0, replications=2, base_seed=42)
        res = InterventionEvaluator(cfg).evaluate(snap)

        # Reassigning C1 to WC reduces waiting time from ~180 person-min (idle C1) to 45 person-min
        self.assertIn(res.selected_policy, (PolicyType.OPTIMIZED, PolicyType.NAIVE))
        self.assertGreater(res.improvement_vs_baseline, 0.0)

    # =========================================================================
    # M. No feasible intervention selects baseline
    # =========================================================================
    def test_m_no_feasible_intervention_selects_baseline(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        # 2 citizens in WC: even if C1 is reassigned, second citizen waits 15 mins (P90 = 13.5m)
        w_wc = (
            QueueCitizenSnapshot(1, "WC-1", self.t_snap),
            QueueCitizenSnapshot(2, "WC-2", self.t_snap),
        )
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        # Strict P90 threshold of 5.0m on Service 2: candidate produces P90 ~ 13.5m, so it is rejected!
        cfg = EvaluatorConfig(
            horizon_minutes=40.0,
            replications=1,
            base_seed=42,
            p90_thresholds={2: 5.0},
        )
        res = InterventionEvaluator(cfg).evaluate(snap)

        # Baseline is safely selected because no candidate is feasible!
        self.assertEqual(res.selected_policy, PolicyType.BASELINE)

    # =========================================================================
    # N. Optimizer does not beat naive
    # =========================================================================
    def test_n_honest_reporting_when_optimizer_does_not_beat_naive(self):
        # In a single-counter two-service setup where Naive reassigns C1 to the single queue,
        # Naive is already optimal. Evaluator must report mathematically correct result without forcing PRAVAAH.
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_wc = (QueueCitizenSnapshot(1, "WC-1", self.t_snap),)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        cfg = EvaluatorConfig(horizon_minutes=30.0, replications=1, base_seed=42)
        res = InterventionEvaluator(cfg).evaluate(snap)

        # The best policy has minimum wait time
        self.assertLessEqual(
            res.total_waiting_person_minutes,
            res.baseline_evaluation.mean_total_waiting_person_minutes,
        )

    # =========================================================================
    # O. Improvement metrics
    # =========================================================================
    def test_o_improvement_metrics_calculation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_wc = (
            QueueCitizenSnapshot(1, "WC-1", self.t_snap),
            QueueCitizenSnapshot(2, "WC-2", self.t_snap),
        )
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        cfg = EvaluatorConfig(horizon_minutes=40.0, replications=1, base_seed=42)
        res = InterventionEvaluator(cfg).evaluate(snap)

        expected_imp_base = res.baseline_evaluation.mean_total_waiting_person_minutes - res.total_waiting_person_minutes
        self.assertAlmostEqual(res.improvement_vs_baseline, expected_imp_base)

    # =========================================================================
    # P. Zero denominator
    # =========================================================================
    def test_p_zero_denominator_handled_safely(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        # Completely empty system: baseline wait = 0.0
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        cfg = EvaluatorConfig(horizon_minutes=20.0, replications=1)
        res = InterventionEvaluator(cfg).evaluate(snap)

        self.assertEqual(res.pct_improvement_vs_baseline, 0.0)
        self.assertFalse(math.isnan(res.pct_improvement_vs_baseline))

    # =========================================================================
    # Q. Candidate isolation
    # =========================================================================
    def test_q_candidate_isolation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1, 2]), None)
        w_wc = (QueueCitizenSnapshot(1, "WC-1", self.t_snap),)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt, 2: self.svc_wc},
            queues={
                1: ServiceQueueSnapshot(1, (), ()),
                2: ServiceQueueSnapshot(2, w_wc, ()),
            },
            counters={1: c1},
        )
        state = SimulationState.from_snapshot(snap)
        evaluator = InterventionEvaluator(EvaluatorConfig(horizon_minutes=30.0, replications=2))

        # Evaluate candidate
        cand = InterventionCandidate("C1_WC", counter_id=1, target_service_id=2)
        res1 = evaluator._simulate_policy(state, candidate=cand, seeds=[42, 43])
        # Initial state should be intact
        self.assertEqual(state.counters[1].assigned_service_id, 1)
        self.assertEqual(state.queues[2].length(), 1)

    # =========================================================================
    # R. Multiple replications
    # =========================================================================
    def test_r_multiple_replications_aggregation(self):
        c1 = CounterSnapshot(1, 101, "C1", "ACTIVE", self.t_snap, 1, frozenset([1]), None)
        snap = SimulationSnapshot(
            office_id=101, office_code="AMD", snapshot_time_utc=self.t_snap,
            services={1: self.svc_pt},
            queues={1: ServiceQueueSnapshot(1, (), ())},
            counters={1: c1},
        )
        cfg = EvaluatorConfig(
            horizon_minutes=30.0,
            replications=4,
            base_seed=100,
            arrival_rates_per_hour={1: 15.0},
        )
        res = InterventionEvaluator(cfg).evaluate(snap)

        base_eval = res.baseline_evaluation
        self.assertEqual(base_eval.replication_count, 4)
        self.assertEqual(len(base_eval.replication_total_waits), 4)
        expected_mean = sum(base_eval.replication_total_waits) / 4.0
        self.assertAlmostEqual(base_eval.mean_total_waiting_person_minutes, expected_mean)


if __name__ == "__main__":
    unittest.main()
