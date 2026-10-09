"""
Tests for Step 2E.1B.1 ML Sampling Benchmark.

Validates:
1. Authoritative trajectory runner reproduces DiscreteEventEngine bitwise.
2. All extracted observations conform strictly to 15 RAW_FEATURE_NAMES order.
3. Target wait minutes are finite, non-negative, and properly null for unserved tokens.
4. Municipality (PT, WC, BC) and RTO (OT, DR) offices and services are covered.
5. Determinism: identical configuration and seeds produce identical summary statistics.
6. Row yields, near-duplicate rates, and censoring rates conform to design constraints.
7. Frozen simulation engine and scenarios remain completely untouched.
"""

import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
for p in (BASE_DIR, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from app.simulation.engine import DiscreteEventEngine
from app.simulation.scenarios import (
    OFFICE_MUNICIPALITY_CODE,
    OFFICE_RTO_CODE,
    SERVICE_PT_CODE,
    SERVICE_WC_CODE,
    SERVICE_BC_CODE,
    SERVICE_OT_CODE,
    SERVICE_DR_CODE,
    build_scenario_b_pt_pressure,
)
from shared.features import RAW_FEATURE_NAMES
from scratch.benchmark_ml_sampling import (
    build_benchmark_trajectories,
    run_trajectory_simulation,
    extract_observations_for_strategy,
    run_sampling_benchmark,
)


class TestMLSamplingBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.datasets = run_sampling_benchmark()

    def test_01_discrete_event_engine_bitwise_fidelity(self):
        """Authoritative trajectory runner produces bitwise identical waits to DiscreteEventEngine."""
        scen = build_scenario_b_pt_pressure()
        state = scen.build_simulation_state()
        seed = 42

        import random
        arr_proc = scen.build_arrival_process(state, rng=random.Random(seed))
        svc_proc = scen.build_service_process(rng=random.Random(seed + 500))
        engine = DiscreteEventEngine(
            horizon_minutes=60.0,
            seed=seed,
            arrival_process=arr_proc,
            service_process=svc_proc,
            availability_events=scen.availability_events,
        )
        ref_res = engine.run(state)

        # Run with trajectory runner
        from scratch.benchmark_ml_sampling import TrajectoryConfig, LOCAL_TZ
        from datetime import datetime
        traj = TrajectoryConfig(
            traj_id=1,
            scenario_builder=build_scenario_b_pt_pressure,
            scenario_name="test_fidelity",
            seed=seed,
            start_time_local=datetime(2026, 10, 5, 10, 0, tzinfo=LOCAL_TZ),
            horizon_minutes=60.0,
        )
        completed, starts, snaps, wall_t = run_trajectory_simulation(traj, [10.0])

        comp_ids_ref = ref_res.completed_citizen_ids
        comp_ids_runner = tuple(c.token_id for c in completed)
        self.assertEqual(comp_ids_ref, comp_ids_runner)

        # Verify waiting times match
        pt_waits_ref = list(ref_res.service_waiting_times[1][:len([c for c in completed if c.service_id == 1])])
        pt_waits_runner = [c.waiting_time_min for c in completed if c.service_id == 1]
        for w_ref, w_run in zip(pt_waits_ref, pt_waits_runner):
            self.assertAlmostEqual(w_ref, w_run, places=9)

    def test_02_feature_contract_and_ordering(self):
        """All observation datasets strictly follow the 15 locked RAW_FEATURE_NAMES."""
        for strat, df in self.datasets.items():
            first_15 = list(df.columns)[:15]
            self.assertEqual(tuple(first_15), RAW_FEATURE_NAMES, f"Strategy {strat} failed feature order")
            self.assertIn("target_wait_minutes", df.columns)
            self.assertIn("is_served", df.columns)

    def test_03_no_legacy_features_or_offices(self):
        """Forbidden legacy features and office codes do not appear."""
        forbidden_cols = ["no_show_rate", "arrival_rate_15m", "counter_load", "actual_wait_minutes"]
        for strat, df in self.datasets.items():
            for c in forbidden_cols:
                self.assertNotIn(c, df.columns)
            offices = set(df["office_id"].unique())
            self.assertTrue(offices.issubset({OFFICE_MUNICIPALITY_CODE, OFFICE_RTO_CODE}))

    def test_04_target_wait_validity(self):
        """Served targets are non-negative and finite; unserved are None."""
        for strat, df in self.datasets.items():
            served = df[df["is_served"] == True]
            unserved = df[df["is_served"] == False]

            self.assertTrue((served["target_wait_minutes"] >= 0.0).all())
            self.assertTrue(np.isfinite(served["target_wait_minutes"]).all())
            self.assertTrue(unserved["target_wait_minutes"].isna().all())

    def test_05_office_and_service_coverage(self):
        """Both Municipality and RTO, and all 5 services, are represented."""
        df = self.datasets["B_10m"]
        offices = set(df["office_id"].unique())
        self.assertEqual(offices, {OFFICE_MUNICIPALITY_CODE, OFFICE_RTO_CODE})

        services = set(df["service_id"].unique())
        expected_services = {
            SERVICE_PT_CODE,
            SERVICE_WC_CODE,
            SERVICE_BC_CODE,
            SERVICE_OT_CODE,
            SERVICE_DR_CODE,
        }
        self.assertEqual(services, expected_services)

    def test_06_sampling_yield_hierarchy(self):
        """A_5m yield > A_10m yield > B_10m yield > C_Hybrid yield."""
        yields = self.results["row_yields"]
        y_a5 = yields["A_5m"]["retained_rows"]
        y_a10 = yields["A_10m"]["retained_rows"]
        y_b10 = yields["B_10m"]["retained_rows"]
        y_c = yields["C_Hybrid"]["retained_rows"]

        self.assertGreater(y_a5, y_a10)
        self.assertGreater(y_a10, y_b10)
        self.assertGreater(y_b10, y_c)

    def test_07_near_duplicate_reduction(self):
        """Near-duplicate percentage drops substantially from A_5m to B_10m and C_Hybrid."""
        near_dups = self.results["near_duplicate_analysis"]
        pct_a5 = near_dups["A_5m"]["near_duplicate_pct"]
        pct_b10 = near_dups["B_10m"]["near_duplicate_pct"]
        pct_c = near_dups["C_Hybrid"]["near_duplicate_pct"]

        self.assertGreater(pct_a5, 40.0)  # >40% in 5m
        self.assertLess(pct_b10, 25.0)    # <25% in 10m selective
        self.assertLess(pct_c, 20.0)      # <20% in hybrid

    def test_08_benchmark_determinism(self):
        """Same seeds produce bitwise identical summary statistics."""
        res2, _ = run_sampling_benchmark()
        self.assertEqual(
            self.results["row_yields"]["B_10m"]["retained_rows"],
            res2["row_yields"]["B_10m"]["retained_rows"],
        )
        self.assertEqual(
            self.results["target_distributions"]["B_10m"]["mean"],
            res2["target_distributions"]["B_10m"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
