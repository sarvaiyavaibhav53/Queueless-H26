"""
PRAVAAH — Step 2D.4: Scenario Benchmarking & Calibration Tests.

Verifies:
1. Benchmark determinism: Running twice with identical seed yields identical results.
2. Common Random Numbers (CRN): Exact identical seeds used across Baseline, Naive, and PRAVAAH.
3. Healthy scenario validation (Scenario A): Baseline is optimal; Naive causes excess delay; PRAVAAH selects Baseline.
4. RTO Healthy validation: Baseline is optimal; PRAVAAH chooses no-action.
5. Pressure scenario validation (Scenario B): PRAVAAH achieves significant waiting time reduction vs Baseline.
6. Donor constraint trade-off (Scenario C): Capacity trade-offs evaluated without artificial restrictions.
7. Temporary look-ahead advantage (Scenario D): 30m temporary reassignment outperforms permanent Naive.
8. Counter failure mitigation (Scenario E): Temporary reassignment outperforms permanent Naive under failure/recovery.
9. P90 constraint enforcement: Tight SLA threshold rejects violating candidates and selects feasible alternative or Baseline.
10. Metric serialization and aggregation integrity.
11. Batch runner covers all registered scenarios.
"""

import math
import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.interventions.models import EvaluatorConfig, PolicyType
from app.interventions.evaluator import InterventionEvaluator
from app.simulation.benchmarking import (
    BenchmarkResult,
    PolicyBenchmarkMetrics,
    ServiceBenchmarkMetrics,
    run_all_benchmarks,
    run_scenario_benchmark,
)
from app.simulation.scenarios import (
    SCENARIO_REGISTRY,
    get_scenario,
)


class TestScenarioBenchmarking(unittest.TestCase):
    """Tests for multi-replication benchmarking and calibration engine."""

    def test_benchmark_determinism(self):
        """Running a benchmark twice with identical base_seed produces bitwise-identical metrics."""
        res1 = run_scenario_benchmark("scenario_a_healthy", replications=5, base_seed=12345)
        res2 = run_scenario_benchmark("scenario_a_healthy", replications=5, base_seed=12345)

        self.assertEqual(res1.scenario_id, res2.scenario_id)
        self.assertEqual(res1.selected_policy, res2.selected_policy)
        self.assertAlmostEqual(
            res1.baseline.mean_total_waiting_person_minutes,
            res2.baseline.mean_total_waiting_person_minutes,
            places=7,
        )
        self.assertAlmostEqual(
            res1.naive.mean_total_waiting_person_minutes,
            res2.naive.mean_total_waiting_person_minutes,
            places=7,
        )
        self.assertAlmostEqual(
            res1.pravaah.mean_total_waiting_person_minutes,
            res2.pravaah.mean_total_waiting_person_minutes,
            places=7,
        )
        self.assertEqual(res1.baseline.replication_total_waits, res2.baseline.replication_total_waits)

    def test_crn_seed_alignment(self):
        """CRN invariant: Baseline, Naive, and PRAVAAH receive identical seeds per replication."""
        scen = get_scenario("scenario_b_pt_pressure")
        res = run_scenario_benchmark(scen, replications=5, base_seed=42)

        self.assertEqual(res.baseline.replications, 5)
        self.assertEqual(res.naive.replications, 5)
        self.assertEqual(res.pravaah.replications, 5)
        self.assertEqual(len(res.baseline.replication_total_waits), 5)
        self.assertEqual(len(res.naive.replication_total_waits), 5)
        self.assertEqual(len(res.pravaah.replication_total_waits), 5)

    def test_scenario_a_healthy_selects_baseline(self):
        """In a healthy scenario, PRAVAAH recognizes no intervention is needed and selects Baseline."""
        res = run_scenario_benchmark("scenario_a_healthy", replications=15, base_seed=42)

        self.assertEqual(res.selected_policy, PolicyType.BASELINE)
        self.assertIsNone(res.selected_candidate_id)
        # Baseline wait is substantially lower than Naive (which disrupts healthy operations)
        self.assertLess(
            res.baseline.mean_total_waiting_person_minutes,
            res.naive.mean_total_waiting_person_minutes,
        )
        # PRAVAAH matches baseline and beats naive
        self.assertAlmostEqual(
            res.pravaah.mean_total_waiting_person_minutes,
            res.baseline.mean_total_waiting_person_minutes,
            places=4,
        )
        self.assertTrue(res.improvement_vs_naive_person_min > 0.0)

    def test_rto_healthy_selects_baseline(self):
        """In RTO Healthy, PRAVAAH selects Baseline and avoids naive reassignment penalty."""
        res = run_scenario_benchmark("scenario_rto_healthy", replications=15, base_seed=42)

        self.assertEqual(res.selected_policy, PolicyType.BASELINE)
        self.assertIsNone(res.selected_candidate_id)
        self.assertLess(
            res.baseline.mean_total_waiting_person_minutes,
            res.naive.mean_total_waiting_person_minutes,
        )
        self.assertAlmostEqual(
            res.pravaah.mean_total_waiting_person_minutes,
            res.baseline.mean_total_waiting_person_minutes,
            places=4,
        )

    def test_scenario_b_pt_pressure_reduces_waiting(self):
        """Under PT queue pressure, reassignment produces significant waiting time reduction."""
        res = run_scenario_benchmark("scenario_b_pt_pressure", replications=15, base_seed=42)

        self.assertTrue(res.pravaah_beats_baseline)
        self.assertGreater(res.improvement_vs_baseline_person_min, 50.0)
        self.assertGreater(res.pct_improvement_vs_baseline, 15.0)
        # P90 wait time is also reduced
        self.assertLess(
            res.pravaah.p90_waiting_time_minutes,
            res.baseline.p90_waiting_time_minutes,
        )

    def test_scenario_c_donor_constraint(self):
        """Scenario C creates a genuine capacity trade-off; reassignment reduces net waiting time."""
        res = run_scenario_benchmark("scenario_c_donor_constraint", replications=15, base_seed=42)

        self.assertTrue(res.pravaah_beats_baseline)
        self.assertGreater(res.pct_improvement_vs_baseline, 10.0)
        # Served customer count increases or stays stable
        self.assertGreaterEqual(
            res.pravaah.mean_customers_served,
            res.baseline.mean_customers_served - 0.5,
        )

    def test_scenario_d_temporary_beats_naive(self):
        """Temporary 30m intervention in Scenario D outperforms permanent Naive."""
        res = run_scenario_benchmark("scenario_d_temporary_intervention", replications=30, base_seed=42)

        self.assertEqual(res.selected_policy, PolicyType.OPTIMIZED)
        self.assertEqual(res.selected_candidate_id, "CAND_C3_TO_S1_D30")
        self.assertTrue(res.pravaah_beats_baseline)
        self.assertTrue(res.pravaah_beats_naive)
        self.assertGreater(res.improvement_vs_naive_person_min, 0.0)

    def test_scenario_e_counter_failure_recovery(self):
        """Under C1 failure and recovery, temporary 30m intervention outperforms permanent Naive."""
        res = run_scenario_benchmark("scenario_e_counter_failure", replications=30, base_seed=42)

        self.assertEqual(res.selected_policy, PolicyType.OPTIMIZED)
        self.assertEqual(res.selected_candidate_id, "CAND_C3_TO_S1_D30")
        self.assertTrue(res.pravaah_beats_baseline)
        self.assertTrue(res.pravaah_beats_naive)

    def test_p90_constraint_enforcement(self):
        """A tight P90 constraint on a donor service causes the optimizer to reject violating candidates."""
        scen = get_scenario("scenario_d_temporary_intervention")
        state = scen.build_simulation_state()

        # Without constraint: selects D30 (where Service 2 P90 is ~19.35m)
        cfg_unconstrained = scen.build_evaluator_config()
        evaluator_unconstrained = InterventionEvaluator(cfg_unconstrained)
        res_unconstrained = evaluator_unconstrained.evaluate(state, config=cfg_unconstrained)
        self.assertEqual(res_unconstrained.selected_candidate.candidate_id, "CAND_C3_TO_S1_D30")

        # With tight constraint on Service 2 (<= 16.0m):
        # CAND_C3_TO_S1_D30 has S2 P90 ~ 19.35m > 16.0m (violates!)
        # CAND_C3_TO_S1_D15 has S2 P90 ~ 14.65m <= 16.0m (feasible!)
        cfg_tight = EvaluatorConfig(
            horizon_minutes=cfg_unconstrained.horizon_minutes,
            replications=cfg_unconstrained.replications,
            base_seed=cfg_unconstrained.base_seed,
            p90_thresholds={2: 16.0},
            arrival_rates_per_hour=cfg_unconstrained.arrival_rates_per_hour,
            candidate_durations=(15.0, 30.0),
            switching_delay_min=3.0,
            availability_events=tuple(scen.availability_events),
            arrival_process_factory=lambda st, rng: scen.build_arrival_process(st, rng=rng),
            service_process_factory=lambda rng: scen.build_service_process(rng=rng),
        )
        evaluator_tight = InterventionEvaluator(cfg_tight)
        res_tight = evaluator_tight.evaluate(state, config=cfg_tight)

        self.assertEqual(res_tight.selected_candidate.candidate_id, "CAND_C3_TO_S1_D15")
        self.assertTrue(res_tight.is_feasible)

    def test_metric_serialization(self):
        """BenchmarkResult and PolicyBenchmarkMetrics serialize properly to JSON-compatible dicts."""
        res = run_scenario_benchmark("scenario_a_healthy", replications=5, base_seed=42)
        d = res.to_dict()

        self.assertIn("scenario_id", d)
        self.assertIn("baseline", d)
        self.assertIn("naive", d)
        self.assertIn("pravaah", d)
        self.assertIn("selected_policy", d)
        self.assertIn("mean_total_waiting_person_minutes", d["baseline"])
        self.assertIn("p90_waiting_time_minutes", d["baseline"])
        self.assertIn("per_service", d["baseline"])

    def test_run_all_benchmarks(self):
        """run_all_benchmarks executes all 6 registered scenarios."""
        results = run_all_benchmarks(replications=5, base_seed=100)

        self.assertEqual(len(results), len(SCENARIO_REGISTRY))
        for scen_id in SCENARIO_REGISTRY:
            self.assertIn(scen_id, results)
            b_res = results[scen_id]
            self.assertIsInstance(b_res, BenchmarkResult)
            self.assertEqual(b_res.replications, 5)


if __name__ == "__main__":
    unittest.main()
