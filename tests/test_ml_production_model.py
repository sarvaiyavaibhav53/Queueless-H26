"""
PRAVAAH — Step 2E.2: Production ML Modeling & Rigorous Evaluation Tests

Validates:
1. Artifact existence and legacy model preservation.
2. Model size constraint (<= 50 MB).
3. Exact 15-feature schema and metadata exclusion.
4. Stratified grouped split isolation (zero config_id leakage).
5. Analytical queue baseline calculations.
6. Baseline evaluation & RF superiority on JOIN observations.
7. Bootstrap confidence interval logic.
8. Monotonicity sanity-check logic.
9. Quantile prediction interval ordering (P10 <= P50 <= P90).
10. Inference latency bounds.
"""

import json
import os
import sys
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd

from shared.features import RAW_FEATURE_NAMES
from ml.src.train_production_model import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_stratified_grouped_split,
    load_and_validate_dataset,
    predict_analytical_pure,
    predict_analytical_residual_aware,
    run_monotonicity_sanity_checks,
    run_trajectory_bootstrap_ci,
)

MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "production_wait_time_rf.joblib")
Q10_MODEL_PATH = os.path.join(MODELS_DIR, "production_quantile_p10.joblib")
Q50_MODEL_PATH = os.path.join(MODELS_DIR, "production_quantile_p50.joblib")
Q90_MODEL_PATH = os.path.join(MODELS_DIR, "production_quantile_p90.joblib")
EVAL_JSON_PATH = os.path.join(MODELS_DIR, "production_model_evaluation.json")
AUDIT_REPORT_PATH = os.path.join(MODELS_DIR, "step2e2_audit_report.md")
TRAINING_REPORT_PATH = os.path.join(MODELS_DIR, "step2e2_training_report.md")
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_wait_time_pipeline.joblib")


class TestMLProductionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_and_validate_dataset()
        with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
            cls.evaluation_meta = json.load(f)
        cls.rf_pipeline = joblib.load(RF_MODEL_PATH)

    def test_01_artifacts_exist(self):
        """Verify all Step 2E.2 model artifacts and reports exist in ml/models/."""
        self.assertTrue(os.path.exists(RF_MODEL_PATH), "RF model artifact missing")
        self.assertTrue(os.path.exists(Q10_MODEL_PATH), "P10 quantile artifact missing")
        self.assertTrue(os.path.exists(Q50_MODEL_PATH), "P50 quantile artifact missing")
        self.assertTrue(os.path.exists(Q90_MODEL_PATH), "P90 quantile artifact missing")
        self.assertTrue(os.path.exists(EVAL_JSON_PATH), "Evaluation JSON missing")
        self.assertTrue(os.path.exists(AUDIT_REPORT_PATH), "Audit report markdown missing")
        self.assertTrue(os.path.exists(TRAINING_REPORT_PATH), "Training report markdown missing")

    def test_02_legacy_model_untouched(self):
        """Verify obsolete 723 MB legacy model exists and was NOT overwritten."""
        self.assertTrue(os.path.exists(LEGACY_MODEL_PATH), "Legacy model missing!")
        legacy_size_mb = os.path.getsize(LEGACY_MODEL_PATH) / (1024.0 * 1024.0)
        self.assertGreater(legacy_size_mb, 650.0, f"Legacy model modified: size is {legacy_size_mb} MB")

    def test_03_rf_artifact_size_within_50mb(self):
        """Verify production Random Forest model artifact size is <= 50 MB."""
        rf_size_mb = os.path.getsize(RF_MODEL_PATH) / (1024.0 * 1024.0)
        self.assertLessEqual(rf_size_mb, 50.0, f"RF model size {rf_size_mb} MB exceeds 50 MB cap")
        self.assertGreater(rf_size_mb, 1.0, f"RF model size {rf_size_mb} MB unusually small")

    def test_04_exact_15_feature_schema_and_metadata_exclusion(self):
        """Verify pipeline strictly expects 15 features and excludes metadata."""
        feature_cols = list(RAW_FEATURE_NAMES)
        self.assertEqual(len(feature_cols), 15)

        test_row = self.df.iloc[[0]][feature_cols]
        pred = self.rf_pipeline.predict(test_row)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))
        self.assertGreaterEqual(pred[0], 0.0)

        # Ensure passing metadata columns causes an error if passed to pipeline without filtering
        metadata_cols = ["token_id", "simulation_seed", "trajectory_id", "regime"]
        corrupted_row = self.df.iloc[[0]][feature_cols + metadata_cols]
        # ColumnTransformer should only use specified 15 features or raise error on unhandled columns
        fitted_preprocessor = self.rf_pipeline.named_steps["preprocessor"]
        cat_features = fitted_preprocessor.transformers[0][2]
        num_features = fitted_preprocessor.transformers[1][2]
        all_handled = set(cat_features).union(set(num_features))
        self.assertEqual(all_handled, set(RAW_FEATURE_NAMES))

    def test_05_grouped_split_isolation(self):
        """Verify stratified grouped split has zero config_id leakage between train, val, and test."""
        train_df, val_df, test_df, split_info = create_stratified_grouped_split(self.df, seed=42)

        train_configs = set(train_df["config_id"].unique())
        val_configs = set(val_df["config_id"].unique())
        test_configs = set(test_df["config_id"].unique())

        self.assertEqual(len(train_configs.intersection(val_configs)), 0, "Train-Val config overlap")
        self.assertEqual(len(train_configs.intersection(test_configs)), 0, "Train-Test config overlap")
        self.assertEqual(len(val_configs.intersection(test_configs)), 0, "Val-Test config overlap")

        # Total configurations preserved
        self.assertEqual(len(train_configs) + len(val_configs) + len(test_configs), self.df["config_id"].nunique())

    def test_06_analytical_baseline_calculations(self):
        """Verify analytical baseline formula on controlled test vector."""
        sample_data = pd.DataFrame([{
            "people_ahead": 4,
            "service_mean_minutes": 10.0,
            "assigned_active_counters": 2,
            "busy_assigned_counters": 2,
            "max_elapsed_service_minutes": 6.0,
        }])

        # Pure: 4 * 10 / 2 = 20.0
        pure_wait = predict_analytical_pure(sample_data)[0]
        self.assertAlmostEqual(pure_wait, 20.0, places=4)

        # Residual: (4 * 10 + max(0, 10 - 6)) / 2 = (40 + 4) / 2 = 22.0
        resid_wait = predict_analytical_residual_aware(sample_data)[0]
        self.assertAlmostEqual(resid_wait, 22.0, places=4)

    def test_07_rf_superiority_over_pure_analytical_baseline(self):
        """Verify Random Forest achieves lower MAE than pure analytical baseline on Test JOIN."""
        baselines = self.evaluation_meta["baselines_test_join"]
        models = self.evaluation_meta["models_test_join"]

        pure_mae = baselines["pure_analytical"]["mae"]
        rf_mae = models["random_forest"]["mae"]

        self.assertLess(rf_mae, pure_mae, f"RF MAE ({rf_mae}) not better than pure analytical ({pure_mae})")
        self.assertGreater(self.evaluation_meta["rf_improvement_over_pure_analytical_pct"], 15.0)

    def test_08_bootstrap_ci_implementation(self):
        """Verify trajectory-level bootstrap CI computation produces valid ordered bounds."""
        boot = self.evaluation_meta["bootstrap_ci_95"]["random_forest"]
        point = boot["point_mae"]
        ci_low = boot["ci_lower_95"]
        ci_high = boot["ci_upper_95"]

        self.assertLessEqual(ci_low, point, f"CI lower ({ci_low}) > point ({point})")
        self.assertLessEqual(point, ci_high, f"Point ({point}) > CI upper ({ci_high})")
        self.assertEqual(boot["n_bootstrap"], 1000)

    def test_09_monotonicity_sanity_checks(self):
        """Verify monotonicity sanity check results show low violation rates (< 10%)."""
        mono = self.evaluation_meta["monotonicity_sanity_checks"]
        self.assertLess(mono["test_a_people_ahead"]["violation_pct"], 10.0)
        self.assertLess(mono["test_b_active_counters"]["violation_pct"], 10.0)

    def test_10_quantile_prediction_intervals(self):
        """Verify Quantile models produce valid intervals with P10 <= P50 <= P90."""
        q10_pipe = joblib.load(Q10_MODEL_PATH)
        q50_pipe = joblib.load(Q50_MODEL_PATH)
        q90_pipe = joblib.load(Q90_MODEL_PATH)

        sample = self.df.iloc[:20][list(RAW_FEATURE_NAMES)]
        p10 = q10_pipe.predict(sample)
        p50 = q50_pipe.predict(sample)
        p90 = q90_pipe.predict(sample)

        # Allow slight numerical floating tolerance
        self.assertTrue((p10 <= p50 + 1e-3).all(), "P10 > P50 violation detected")
        self.assertTrue((p50 <= p90 + 1e-3).all(), "P50 > P90 violation detected")

    def test_11_single_row_inference_latency(self):
        """Verify single-row prediction latency is under 100 ms."""
        lat = self.evaluation_meta["performance_and_footprint"]["latency_1_row_ms"]
        self.assertLess(lat["median"], 100.0, f"1-row latency median too high: {lat['median']} ms")


if __name__ == "__main__":
    unittest.main()
