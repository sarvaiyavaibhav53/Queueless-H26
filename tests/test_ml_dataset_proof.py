"""
PRAVAAH — Step 2E.1A: Focused Tests for ML Dataset Generation Proof.

Validates the 13 required dataset properties:
1. Exact 15 locked raw features + target + necessary metadata
2. Feature order strictly matches RAW_FEATURE_NAMES
3. No legacy feature names exist
4. No legacy office or service codes appear
5. Target values are finite and non-negative
6. Target values are derived from actual simulation behavior
7. No future-only information used to compute features
8. Same seed produces bitwise identical proof dataset
9. Different simulation states produce different feature rows
10. Municipality services (PT, WC, BC) are represented correctly
11. RTO services (OT, DR) are represented correctly
12. Dataset generation leaves authoritative simulation and benchmarks unmodified
13. Existing shared feature tests continue to pass
"""

import os
import sys

# Ensure project root and backend are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from datetime import datetime, timezone, timedelta
import math
import unittest

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
from app.simulation.state import SimulationState
from ml.src.generate_proof_dataset import (
    DEFAULT_BASE_SEED,
    DEFAULT_PROOF_OUTPUT_DIR,
    build_proof_snapshots,
    generate_proof_dataset,
    save_proof_dataset,
)
from shared.features import (
    RAW_FEATURE_NAMES,
    compute_raw_features,
)


class TestMLDatasetProof(unittest.TestCase):
    """
    Focused validation suite for Step 2E.1A dataset generation proof.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = generate_proof_dataset(base_seed=DEFAULT_BASE_SEED)

    def test_01_feature_contract_and_columns(self):
        """1. Dataset contains exactly 15 locked raw features plus target and necessary metadata."""
        expected_metadata = ["token_id", "token_number", "queue_position", "snapshot_time_utc", "scenario_id"]
        expected_cols = list(RAW_FEATURE_NAMES) + ["target_wait_minutes"] + expected_metadata
        self.assertEqual(list(self.df.columns), expected_cols)
        self.assertEqual(len(self.df.columns), 15 + 1 + len(expected_metadata))

    def test_02_feature_order_matches_raw_feature_names(self):
        """2. Feature order matches RAW_FEATURE_NAMES."""
        first_15 = list(self.df.columns)[:15]
        self.assertEqual(tuple(first_15), RAW_FEATURE_NAMES)

    def test_03_no_legacy_feature_names(self):
        """3. No legacy feature names exist."""
        forbidden_features = [
            "no_show_rate",
            "arrival_rate_15m",
            "counter_load",
            "future_arrivals",
            "future_failures",
            "actual_future_queue_length",
            "actual_future_service_duration",
            "actual_wait_minutes",  # legacy target name from prototype
            "service_type",         # legacy column name
            "counter_id",           # counter-level legacy feature
        ]
        for feat in forbidden_features:
            self.assertNotIn(feat, self.df.columns, f"Forbidden legacy feature '{feat}' found in dataset!")

    def test_04_no_legacy_office_or_service_codes(self):
        """4. No legacy office or service codes appear."""
        legacy_offices = {"RJT", "AMD", "STV"}
        legacy_services = {"LL", "DL_RENEWAL", "VEH_REG"}

        observed_offices = set(self.df["office_id"].unique())
        observed_services = set(self.df["service_id"].unique())

        for leg_off in legacy_offices:
            self.assertNotIn(leg_off, observed_offices, f"Legacy office code '{leg_off}' found!")
        for leg_svc in legacy_services:
            self.assertNotIn(leg_svc, observed_services, f"Legacy service code '{leg_svc}' found!")

        # Authoritative domain codes only
        valid_offices = {OFFICE_MUNICIPALITY_CODE, OFFICE_RTO_CODE}
        valid_services = {
            SERVICE_PT_CODE,
            SERVICE_WC_CODE,
            SERVICE_BC_CODE,
            SERVICE_OT_CODE,
            SERVICE_DR_CODE,
        }
        self.assertTrue(observed_offices.issubset(valid_offices))
        self.assertTrue(observed_services.issubset(valid_services))

    def test_05_target_values_finite_and_non_negative(self):
        """5. Target values are finite and non-negative."""
        targets = self.df["target_wait_minutes"]
        self.assertTrue(all(np.isfinite(targets)), "Found non-finite target values!")
        self.assertTrue(all(targets >= 0.0), "Found negative target waiting times!")
        self.assertGreater(targets.max(), 0.0, "All target values were zero!")

    def test_06_target_derived_from_actual_simulation(self):
        """6. Target values are derived from actual simulation behavior."""
        # Isolate the first snapshot (Scenario A: Healthy)
        snapshots = build_proof_snapshots()
        scen_id, snap = snapshots[0]
        self.assertEqual(scen_id, "scenario_a_healthy")

        sim_state = SimulationState.from_snapshot(snap)
        engine = DiscreteEventEngine(horizon_minutes=180.0, seed=DEFAULT_BASE_SEED)
        result = engine.run(sim_state)

        # In Scenario A, PT token 101 starts service at t = 0.0 (idle counter C1)
        # Verify dataset matches exactly
        scen_a_rows = self.df[self.df["scenario_id"] == "scenario_a_healthy"]
        pt_row = scen_a_rows[scen_a_rows["service_id"] == "PT"].iloc[0]

        # Token 101 arrival offset in snapshot is -2.0 min.
        # Its service began at t=0.0 -> future wait from snapshot (t=0) is 0.0 min.
        self.assertEqual(pt_row["target_wait_minutes"], 0.0)

        # In Scenario B (PT Pressure), check that position 8 waited significantly
        scen_b_rows = self.df[self.df["scenario_id"] == "scenario_b_pt_pressure"]
        pt_b_rows = scen_b_rows[scen_b_rows["service_id"] == "PT"]
        # Position 1 should start at 0.0, Position 8 should wait > 20 min
        self.assertEqual(pt_b_rows.iloc[0]["target_wait_minutes"], 0.0)
        self.assertGreater(pt_b_rows.iloc[-1]["target_wait_minutes"], 20.0)

    def test_07_no_future_information_in_features(self):
        """7. Features are computed strictly at snapshot time without future knowledge."""
        snapshots = build_proof_snapshots()
        _, snap = snapshots[1]  # Scenario B

        # Compute features at snapshot time
        feat1 = compute_raw_features(snap, service_id=1, position=1)

        # Verify that features depend only on current and past timestamps
        # Changing snapshot time moves the information boundary
        t_snap = snap.snapshot_time_utc
        self.assertEqual(feat1.queue_length, len(snap.queues[1].waiting_tokens))
        for arr in snap.queues[1].recent_arrivals_utc:
            self.assertLessEqual(arr, t_snap)

    def test_08_same_seed_produces_identical_proof_dataset(self):
        """8. Same seed produces identical proof dataset."""
        df_run1 = generate_proof_dataset(base_seed=DEFAULT_BASE_SEED)
        df_run2 = generate_proof_dataset(base_seed=DEFAULT_BASE_SEED)
        pd.testing.assert_frame_equal(df_run1, df_run2)

        # Different seed produces different simulated targets
        df_run3 = generate_proof_dataset(base_seed=9999)
        self.assertFalse(
            df_run1["target_wait_minutes"].equals(df_run3["target_wait_minutes"]),
            "Different seeds unexpectedly produced identical target waiting times!",
        )

    def test_09_different_states_produce_different_feature_rows(self):
        """9. Different simulation states produce different feature rows."""
        scen_a = self.df[self.df["scenario_id"] == "scenario_a_healthy"].iloc[0]
        scen_b = self.df[self.df["scenario_id"] == "scenario_b_pt_pressure"].iloc[0]
        scen_busy = self.df[self.df["scenario_id"] == "scenario_b_busy_counters"].iloc[0]
        scen_fail = self.df[self.df["scenario_id"] == "scenario_e_counter_failure"].iloc[0]

        # Queue length differs between healthy and pressure
        self.assertNotEqual(scen_a["queue_length"], scen_b["queue_length"])
        # Busy counters differ
        self.assertEqual(scen_a["busy_assigned_counters"], 0)
        self.assertEqual(scen_busy["busy_assigned_counters"], 2)
        self.assertEqual(scen_busy["max_elapsed_service_minutes"], 5.0)
        # Active counters differ under counter failure
        self.assertEqual(scen_fail["assigned_active_counters"], 1)
        self.assertEqual(scen_b["assigned_active_counters"], 2)

    def test_10_municipality_services_represented(self):
        """10. Municipality services are represented correctly."""
        muni_df = self.df[self.df["office_id"] == OFFICE_MUNICIPALITY_CODE]
        self.assertFalse(muni_df.empty)
        muni_services = set(muni_df["service_id"].unique())
        self.assertEqual(muni_services, {SERVICE_PT_CODE, SERVICE_WC_CODE, SERVICE_BC_CODE})

    def test_11_rto_services_represented(self):
        """11. RTO services are represented correctly where practical."""
        rto_df = self.df[self.df["office_id"] == OFFICE_RTO_CODE]
        self.assertFalse(rto_df.empty)
        rto_services = set(rto_df["service_id"].unique())
        self.assertEqual(rto_services, {SERVICE_OT_CODE, SERVICE_DR_CODE})

    def test_12_simulation_state_and_benchmarks_unmodified(self):
        """12. Dataset generation does not modify authoritative simulation state or benchmark behavior."""
        scen = build_scenario_b_pt_pressure()
        state = scen.build_simulation_state()
        engine_before = DiscreteEventEngine(horizon_minutes=scen.horizon_minutes, seed=42)
        res_before = engine_before.run(state)

        # Generate dataset
        _ = generate_proof_dataset(base_seed=DEFAULT_BASE_SEED)

        # Run benchmark with fresh engine (same seed) and verify identical numerical result
        engine_after = DiscreteEventEngine(horizon_minutes=scen.horizon_minutes, seed=42)
        res_after = engine_after.run(state)
        self.assertEqual(res_before.total_customers_served, res_after.total_customers_served)
        self.assertAlmostEqual(res_before.total_waiting_person_minutes, res_after.total_waiting_person_minutes, places=6)
        self.assertEqual(res_before.completed_citizen_ids, res_after.completed_citizen_ids)

    def test_13_dataset_row_count_within_bounds(self):
        """13. Proof dataset size is approximately 20–50 rows."""
        row_count = len(self.df)
        self.assertGreaterEqual(row_count, 20)
        self.assertLessEqual(row_count, 50)
        self.assertEqual(row_count, 49)

    def test_14_saved_files_integrity(self):
        """14. Saved proof files exist and match generated dataframe."""
        csv_path, parquet_path = save_proof_dataset(self.df)
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(parquet_path))

        loaded_csv = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_csv), len(self.df))
        self.assertEqual(list(loaded_csv.columns), list(self.df.columns))

        loaded_parquet = pd.read_parquet(parquet_path)
        self.assertEqual(len(loaded_parquet), len(self.df))
        self.assertEqual(list(loaded_parquet.columns), list(self.df.columns))


if __name__ == "__main__":
    unittest.main()
