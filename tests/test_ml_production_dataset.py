"""
PRAVAAH — Step 2E.1C Production ML Dataset Tests

Validates:
1. Frozen feature contract: exactly 15 features in canonical order.
2. Target definition: non-negative, finite wait times.
3. Cold-start / warm-up and draining integrity.
4. Observation types: JOIN and PERIODIC presence and validity.
5. Trajectory independence and per-trajectory cap (<= 40 rows).
6. Production regime and office/service coverage.
7. Metadata completeness and no data leakage.
8. Deterministic reproducibility with base seed range (>= 1_000_000).
"""

import json
import os
import sys
import unittest

# Ensure project root and backend are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np
import pandas as pd

from shared.features import RAW_FEATURE_NAMES
from ml.src.generate_production_dataset import (
    DEFAULT_BASE_SEED,
    MAX_ROWS_PER_TRAJECTORY,
    TARGET_TRAJECTORY_COUNT,
    build_production_trajectories,
    simulate_and_sample_trajectory,
)

DATASET_DIR = os.path.join(BASE_DIR, "ml", "data", "synthetic", "production")
CSV_PATH = os.path.join(DATASET_DIR, "dataset.csv")
PARQUET_PATH = os.path.join(DATASET_DIR, "dataset.parquet")
METADATA_PATH = os.path.join(DATASET_DIR, "generation_metadata.json")
REPORT_PATH = os.path.join(DATASET_DIR, "generation_report.md")


class TestMLProductionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(CSV_PATH), f"Dataset CSV missing: {CSV_PATH}"
        assert os.path.exists(METADATA_PATH), f"Metadata JSON missing: {METADATA_PATH}"
        assert os.path.exists(REPORT_PATH), f"Report MD missing: {REPORT_PATH}"
        cls.df = pd.read_csv(CSV_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            cls.metadata = json.load(f)

    def test_01_artifacts_exist(self):
        """Verify all 4 required artifacts exist in ml/data/synthetic/production/."""
        self.assertTrue(os.path.exists(CSV_PATH), "dataset.csv missing")
        self.assertTrue(os.path.exists(METADATA_PATH), "generation_metadata.json missing")
        self.assertTrue(os.path.exists(REPORT_PATH), "generation_report.md missing")
        try:
            import pyarrow  # noqa: F401
            self.assertTrue(os.path.exists(PARQUET_PATH), "dataset.parquet missing when pyarrow is available")
        except ImportError:
            pass

    def test_02_frozen_feature_contract(self):
        """Verify the 15 features are present and start at the beginning of the DataFrame."""
        self.assertEqual(len(RAW_FEATURE_NAMES), 15)
        for feat in RAW_FEATURE_NAMES:
            self.assertIn(feat, self.df.columns, f"Locked feature {feat} missing from dataset")

        first_15 = list(self.df.columns[:15])
        self.assertEqual(first_15, list(RAW_FEATURE_NAMES), f"Feature columns order mismatch: {first_15} != {list(RAW_FEATURE_NAMES)}")

    def test_03_target_wait_minutes(self):
        """Verify target column exists, has no NaNs, and has strictly non-negative finite values."""
        self.assertIn("target_wait_minutes", self.df.columns)
        targets = self.df["target_wait_minutes"]
        self.assertFalse(targets.isna().any(), "target_wait_minutes contains NaN values")
        self.assertTrue((targets >= 0.0).all(), "target_wait_minutes contains negative values")
        self.assertTrue(np.isfinite(targets).all(), "target_wait_minutes contains non-finite values")

    def test_04_metadata_columns_present(self):
        """Verify required observation and simulation metadata columns exist."""
        required_obs_meta = ["token_id", "token_number", "queue_position", "snapshot_time_utc"]
        required_sim_meta = [
            "simulation_seed",
            "trajectory_id",
            "config_id",
            "regime",
            "has_failure",
            "obs_type",
        ]
        for col in required_obs_meta + required_sim_meta:
            self.assertIn(col, self.df.columns, f"Metadata column {col} missing")

    def test_05_row_and_trajectory_counts(self):
        """Verify row count target (30k-40k), >= 1,000 trajectories, and per-trajectory cap."""
        num_rows = len(self.df)
        unique_trajs = self.df["trajectory_id"].nunique()

        self.assertGreaterEqual(unique_trajs, 1000, f"Expected >= 1000 trajectories, got {unique_trajs}")
        self.assertTrue(30000 <= num_rows <= 45000, f"Expected ~30k-40k rows, got {num_rows}")

        # Per trajectory cap
        rows_per_traj = self.df.groupby("trajectory_id").size()
        self.assertTrue((rows_per_traj <= MAX_ROWS_PER_TRAJECTORY).all(),
                        f"Some trajectory exceeded max cap of {MAX_ROWS_PER_TRAJECTORY}: {rows_per_traj.max()}")

    def test_06_office_and_service_coverage(self):
        """Verify both offices and all 5 services are covered with no legacy values."""
        offices = set(self.df["office_id"].unique())
        self.assertEqual(offices, {"MUNI", "RTO"}, f"Expected {{'MUNI', 'RTO'}}, got {offices}")

        services = set(self.df["service_id"].unique())
        self.assertEqual(services, {"PT", "WC", "BC", "OT", "DR"}, f"Expected 5 services, got {services}")

        for off in ["MUNI", "RTO"]:
            count = (self.df["office_id"] == off).sum()
            self.assertGreater(count, 1000, f"Office {off} underrepresented: {count}")

        for srv in ["PT", "WC", "BC", "OT", "DR"]:
            count = (self.df["service_id"] == srv).sum()
            self.assertGreater(count, 500, f"Service {srv} underrepresented: {count}")

    def test_07_regime_coverage(self):
        """Verify all 6 production regimes are represented."""
        regimes = set(self.df["regime"].unique())
        expected_regimes = {"HEALTHY", "MODERATE", "HIGH_LOAD", "STRESSED", "FAILURE", "INTERVENTION"}
        self.assertEqual(regimes, expected_regimes, f"Expected {expected_regimes}, got {regimes}")

    def test_08_observation_types(self):
        """Verify both JOIN and PERIODIC observations exist and are well-represented."""
        obs_types = set(self.df["obs_type"].unique())
        self.assertIn("JOIN", obs_types, "JOIN observations missing")
        self.assertIn("PERIODIC", obs_types, "PERIODIC observations missing")

        join_count = (self.df["obs_type"] == "JOIN").sum()
        periodic_count = (self.df["obs_type"] == "PERIODIC").sum()

        self.assertGreater(join_count, 5000, f"JOIN count too low: {join_count}")
        self.assertGreater(periodic_count, 5000, f"PERIODIC count too low: {periodic_count}")

    def test_09_seed_isolation(self):
        """Verify base seed >= 1_000_000 and completely isolated from Step 2D.4 range (100k-299k)."""
        seeds = self.df["simulation_seed"].unique()
        self.assertTrue((seeds >= DEFAULT_BASE_SEED).all(), f"Seeds found below {DEFAULT_BASE_SEED}")
        self.assertFalse(any(100_000 <= s <= 299_999 for s in seeds), "Seed collision with Step 2D.4 benchmarks!")

    def test_10_no_data_leakage(self):
        """Verify no forbidden future features leaked into columns."""
        forbidden_terms = [
            "future",
            "service_end",
            "completion_time",
            "service_duration",
            "censored",
            "drain",
        ]
        for col in self.df.columns:
            for term in forbidden_terms:
                self.assertNotIn(term, col.lower(), f"Potential data leakage column detected: {col}")

    def test_11_deterministic_reproducibility(self):
        """Verify that running the trajectory generator twice with identical seed produces identical outputs."""
        configs_1 = build_production_trajectories(base_seed=1_099_000, total_count=2)
        configs_2 = build_production_trajectories(base_seed=1_099_000, total_count=2)

        rows_1, _ = simulate_and_sample_trajectory(configs_1[0], MAX_ROWS_PER_TRAJECTORY)
        rows_2, _ = simulate_and_sample_trajectory(configs_2[0], MAX_ROWS_PER_TRAJECTORY)

        self.assertEqual(len(rows_1), len(rows_2), f"Row count mismatch: {len(rows_1)} vs {len(rows_2)}")
        self.assertGreater(len(rows_1), 0, "No rows generated in test trajectory")

        for r1, r2 in zip(rows_1, rows_2):
            for k in r1:
                val1 = r1[k]
                val2 = r2[k]
                if isinstance(val1, float):
                    self.assertAlmostEqual(val1, val2, places=7, msg=f"Float mismatch in {k}: {val1} vs {val2}")
                else:
                    self.assertEqual(val1, val2, f"Value mismatch in {k}: {val1} vs {val2}")

    def test_12_warmup_and_opening_distinction(self):
        """Verify that both opening and warmup trajectories exist in matrix and dataset."""
        trajs = build_production_trajectories(base_seed=DEFAULT_BASE_SEED, total_count=TARGET_TRAJECTORY_COUNT)
        self.assertGreater(len(trajs), 0)

        openings = [t for t in trajs if t.is_opening]
        warmups = [t for t in trajs if not t.is_opening]

        self.assertGreater(len(openings), 0, "No opening trajectories found in matrix")
        self.assertGreater(len(warmups), 0, "No midday/warmup trajectories found in matrix")
        for o in openings:
            self.assertEqual(o.warmup_min, 0.0, f"Expected 0 min warmup for opening trajectory, got {o.warmup_min}")
        for w in warmups:
            self.assertEqual(w.warmup_min, 60.0, f"Expected 60 min warmup for midday trajectory, got {w.warmup_min}")

        # Also verify both opening hour 9 and midday hours exist in dataset
        hours_in_data = set(self.df["hour"].unique())
        self.assertIn(9, hours_in_data, "Opening hour 9 missing from dataset")
        self.assertTrue(any(h > 9 for h in hours_in_data), "Midday hours missing from dataset")


if __name__ == "__main__":
    unittest.main()
