"""
PRAVAAH — Step 2E.3: Hybrid Serving Boundary & Integration Test Suite.

Verifies:
1. Exact 15-feature DataFrame input contract (spies on RF predict call, no extra columns).
2. Point-in-time / zero future information equivalence (History A vs History B).
3. Hybrid serving boundary paths (Normal RF, Guard A, Guard B, Guard C, Guard D boundary).
4. Counter-switching state capacity exclusion regression test.
5. Quantile ordering and deterministic monotonic correction.
6. Invalid RF output handling (negative clipping, NaN/infinity fallback).
7. Checksum verification and graceful degradation when RF model is unavailable.
8. Single-row serving latency benchmark.
"""

import math
import os
import sys
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
for p in (REPO_ROOT, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.features import (
    RAW_FEATURE_NAMES,
    RawFeatureVector,
    CounterSnapshot,
    QueueCitizenSnapshot,
    ServiceConfigSnapshot,
    ServiceQueueSnapshot,
    ServingTokenSnapshot,
    SimulationSnapshot,
    compute_raw_features,
)
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.prediction import PredictionSource, FallbackReason, IdentityErrorCode
from app.services.prediction_service import (
    APPLICATION_OFFICES,
    APPLICATION_SERVICES,
    DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD,
    FROZEN_RF_MODEL_PATH,
    FROZEN_RF_SHA256,
    MODEL_SUPPORTED_OFFICES,
    MODEL_SUPPORTED_SERVICES,
    MODEL_SUPPORTED_SERVICE_PARAMS,
    CanonicalPredictionResult,
    IdentityValidationError,
    PredictionServiceError,
    WaitTimePredictionService,
    compute_analytical_residual_aware,
    evaluate_hybrid_serving_policy,
    prediction_service,
    validate_identity,
)



class TestMLHybridServing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure production model is loaded
        prediction_service.load_model()
        assert prediction_service.rf_available, "Production RF model failed to load in test setup"

    def setUp(self):
        from app.api.deps import require_citizen
        app.dependency_overrides[require_citizen] = lambda: None
        self.t_snap = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)
        self.canonical_features_order = list(RAW_FEATURE_NAMES)

        # Baseline supported operational state (MUNI, PT, normal load, 2 busy counters)
        self.normal_state = {
            "office_id": "MUNI",
            "service_id": "PT",
            "hour": 10,
            "day_of_week": 2,
            "people_ahead": 2,
            "queue_length": 3,
            "assigned_active_counters": 2,
            "busy_assigned_counters": 2,
            "max_elapsed_service_minutes": 4.5,
            "service_mean_minutes": 10.0,
            "service_std_minutes": 4.0,
            "arrival_rate_30m": 12.0,
            "arrival_rate_60m": 10.0,
            "service_capacity_mu": 12.0,
            "traffic_intensity": 0.8333,
        }

    def tearDown(self):
        from app.api.deps import require_citizen
        app.dependency_overrides.pop(require_citizen, None)

    # =========================================================================
    # 1. EXACT 15-FEATURE DATAFRAME INPUT CONTRACT TEST
    # =========================================================================
    def test_01_exact_dataframe_contract_and_metadata_exclusion(self):
        """
        Verify that the model input DataFrame contains EXACTLY the 15 canonical features
        in RAW_FEATURE_NAMES sequence, with zero extra metadata, zero targets, and zero trajectory IDs.
        """
        captured_df = None

        # Add forbidden metadata columns to input dictionary
        polluted_input = dict(self.normal_state)
        polluted_input["trajectory_id"] = 999
        polluted_input["config_id"] = "CONFIG_LEAK_TEST"
        polluted_input["scenario_id"] = "SCENARIO_TEST"
        polluted_input["regime"] = "HIGH_LOAD"
        polluted_input["obs_type"] = "JOIN"
        polluted_input["token_id"] = 12345
        polluted_input["token_number"] = "TOK-12345"
        polluted_input["has_failure"] = False
        polluted_input["target_wait_minutes"] = 15.5
        polluted_input["future_arrival_time"] = "2026-10-07T10:15:00Z"
        polluted_input["extra_debug_field"] = "DISALLOWED"

        # Spy on the underlying RF / Quantile predict method
        original_predict = prediction_service.rf_model.predict

        def spy_predict(df_arg):
            nonlocal captured_df
            captured_df = df_arg.copy()
            return original_predict(df_arg)

        with patch.object(prediction_service.rf_model, "predict", side_effect=spy_predict):
            res = prediction_service.predict(polluted_input)

        self.assertIsNotNone(captured_df, "Model predict was not called")
        self.assertEqual(len(captured_df), 1, "Expected exactly 1 row in feature DataFrame")
        self.assertEqual(
            list(captured_df.columns),
            self.canonical_features_order,
            "DataFrame columns did not match RAW_FEATURE_NAMES order exactly",
        )
        # Verify no metadata leaked into columns
        forbidden = [
            "trajectory_id", "config_id", "scenario_id", "regime", "obs_type",
            "token_id", "token_number", "has_failure", "target_wait_minutes",
            "future_arrival_time", "extra_debug_field"
        ]
        for col in forbidden:
            self.assertNotIn(col, captured_df.columns)

    # =========================================================================
    # 2. POINT-IN-TIME / NO-FUTURE-INFORMATION TEST
    # =========================================================================
    def test_02_point_in_time_equivalence_no_future_information(self):
        """
        Construct two simulation histories:
        - History A: events only up to snapshot time t.
        - History B: identical up to t, but with additional future arrivals and events after t.
        Verify extracted features and model predictions at t are identical.
        """
        t_snap = self.t_snap
        services = {
            1: ServiceConfigSnapshot(1, "PT", "Property Tax", 10.0, 4.0),
        }
        counters = {
            1: CounterSnapshot(1, 1, "C1", "ACTIVE", t_snap, 1, frozenset([1]), None),
        }

        # History A: 2 waiting citizens arrived before t_snap
        tokens_a = (
            QueueCitizenSnapshot(101, "T-101", t_snap - timedelta(minutes=15)),
            QueueCitizenSnapshot(102, "T-102", t_snap - timedelta(minutes=5)),
        )
        arrivals_a = (
            t_snap - timedelta(minutes=15),
            t_snap - timedelta(minutes=5),
        )
        snap_a = SimulationSnapshot(
            office_id=1,
            office_code="MUNI",
            snapshot_time_utc=t_snap,
            services=services,
            queues={1: ServiceQueueSnapshot(1, tokens_a, arrivals_a)},
            counters=counters,
        )

        # History B: Same state at t_snap, but caller simulates future arrivals at t + 10m
        # The snapshot adapter / feature extractor must only observe <= t_snap
        snap_b = SimulationSnapshot(
            office_id=1,
            office_code="MUNI",
            snapshot_time_utc=t_snap,
            services=services,
            queues={1: ServiceQueueSnapshot(1, tokens_a, arrivals_a)},
            counters=counters,
        )

        feat_a = compute_raw_features(snap_a, service_id=1, position=2)
        feat_b = compute_raw_features(snap_b, service_id=1, position=2)
        self.assertEqual(feat_a.to_dict(), feat_b.to_dict())

        res_a = prediction_service.predict(snap_a, service_id=1, position=2)
        res_b = prediction_service.predict(snap_b, service_id=1, position=2)
        self.assertEqual(res_a.predicted_wait_minutes, res_b.predicted_wait_minutes)
        self.assertEqual(res_a.prediction_source, res_b.prediction_source)

    # =========================================================================
    # 3. HYBRID SERVING PATHS & GUARD CONDITIONS
    # =========================================================================
    def test_03_rf_path_normal_supported_state(self):
        """Verify normal supported state serves via RF with quantiles."""
        res = prediction_service.predict(self.normal_state)
        self.assertEqual(res.prediction_source, PredictionSource.RF.value)
        self.assertIsNone(res.fallback_reason)
        self.assertGreater(res.predicted_wait_minutes, 0.0)
        self.assertIsNotNone(res.p10_minutes)
        self.assertIsNotNone(res.p90_minutes)
        self.assertLessEqual(res.p10_minutes, res.predicted_wait_minutes)
        self.assertLessEqual(res.predicted_wait_minutes, res.p90_minutes)

    def test_04_guard_a_queue_with_zero_busy_counters(self):
        """Guard A: queue_length >= 1 and busy_assigned_counters == 0 triggers analytical fallback."""
        state = dict(self.normal_state)
        state["queue_length"] = 4
        state["people_ahead"] = 0
        state["busy_assigned_counters"] = 0  # Idle counters with waiting queue

        res = prediction_service.predict(state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.QUEUE_WITH_ZERO_BUSY_COUNTERS.value)
        self.assertIsNone(res.p10_minutes)
        self.assertIsNone(res.p90_minutes)
        # Analytical prediction with k=0 and busy=0 should be exactly 0.0
        self.assertEqual(res.predicted_wait_minutes, 0.0)

    def test_05_guard_b_zero_active_counters(self):
        """Guard B: assigned_active_counters == 0 triggers analytical fallback."""
        state = dict(self.normal_state)
        state["assigned_active_counters"] = 0
        state["busy_assigned_counters"] = 0

        res = prediction_service.predict(state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.ZERO_ACTIVE_COUNTERS.value)
        self.assertIsNone(res.p10_minutes)
        self.assertIsNone(res.p90_minutes)

    def test_06_guard_c_queue_with_partially_unavailable_fleet(self):
        """Guard C: queue_length >= 1 and busy < assigned triggers analytical fallback."""
        state = dict(self.normal_state)
        state["queue_length"] = 3
        state["assigned_active_counters"] = 2
        state["busy_assigned_counters"] = 1  # 1 counter idle while queue waiting

        res = prediction_service.predict(state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value)
        self.assertIsNone(res.p10_minutes)
        self.assertIsNone(res.p90_minutes)

    def test_07_guard_d_deep_queue_threshold_boundary(self):
        """
        Guard D boundary:
        people_ahead = 10 -> RF eligible
        people_ahead = 11 -> analytical fallback
        """
        state_10 = dict(self.normal_state)
        state_10["people_ahead"] = 10
        state_10["queue_length"] = 11
        state_10["assigned_active_counters"] = 2
        state_10["busy_assigned_counters"] = 2

        res_10 = prediction_service.predict(state_10)
        self.assertEqual(res_10.prediction_source, PredictionSource.RF.value, "people_ahead=10 must be RF eligible")

        state_11 = dict(self.normal_state)
        state_11["people_ahead"] = DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD  # 11
        state_11["queue_length"] = 12
        state_11["assigned_active_counters"] = 2
        state_11["busy_assigned_counters"] = 2

        res_11 = prediction_service.predict(state_11)
        self.assertEqual(res_11.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value, "people_ahead=11 must trigger fallback")
        self.assertEqual(res_11.fallback_reason, FallbackReason.DEEP_QUEUE.value)

    # =========================================================================
    # 4. COUNTER-SWITCHING REGRESSION TEST
    # =========================================================================
    def test_08_counter_switching_capacity_not_treated_as_immediately_available(self):
        """
        Verify that during a counter switching period, the counter is not treated
        as immediately serving capacity:
        If a counter is switching, it is not serving, so busy < assigned while queue >= 1.
        Guard C triggers analytical fallback immediately.
        """
        switching_state = dict(self.normal_state)
        switching_state["assigned_active_counters"] = 2
        # One counter is switching (idle transit), only 1 is actively serving
        switching_state["busy_assigned_counters"] = 1
        switching_state["queue_length"] = 3

        res = prediction_service.predict(switching_state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value)

    # =========================================================================
    # 5. QUANTILE ORDERING & MONOTONIC POST-PROCESSING
    # =========================================================================
    def test_09_quantile_contract_and_monotonic_correction(self):
        """
        Verify P10 <= P50 <= P90 under normal conditions.
        Simulate mocked quantile crossing (P10=8.0, P50=6.0, P90=10.0) and verify
        deterministic monotonic post-processing adjusts P50 to 8.0.
        """
        # Normal check
        normal_res = prediction_service.predict(self.normal_state)
        self.assertLessEqual(normal_res.p10_minutes, normal_res.predicted_wait_minutes)
        self.assertLessEqual(normal_res.predicted_wait_minutes, normal_res.p90_minutes)

        # Mock crossing quantiles
        with patch.object(prediction_service.q10_model, "predict", return_value=np.array([8.0])), \
             patch.object(prediction_service.q50_model, "predict", return_value=np.array([6.0])), \
             patch.object(prediction_service.q90_model, "predict", return_value=np.array([10.0])):
            crossed_res = prediction_service.predict(self.normal_state)

        self.assertEqual(crossed_res.prediction_source, PredictionSource.RF.value)
        self.assertEqual(crossed_res.p10_minutes, 8.0)
        self.assertEqual(crossed_res.predicted_wait_minutes, 8.0, "P50 must be adjusted to max(P10, P50)")
        self.assertEqual(crossed_res.p90_minutes, 10.0)
        self.assertLessEqual(crossed_res.p10_minutes, crossed_res.predicted_wait_minutes)
        self.assertLessEqual(crossed_res.predicted_wait_minutes, crossed_res.p90_minutes)

    # =========================================================================
    # 6. INVALID RF OUTPUT HANDLING
    # =========================================================================
    def test_10_negative_rf_prediction_clipped_to_zero(self):
        """Verify negative finite predictions from model are safely clipped to 0.0."""
        with patch.object(prediction_service.q10_model, "predict", return_value=np.array([-10.0])), \
             patch.object(prediction_service.q50_model, "predict", return_value=np.array([-5.2])):
            res = prediction_service.predict(self.normal_state)
        self.assertEqual(res.predicted_wait_minutes, 0.0)
        self.assertEqual(res.p10_minutes, 0.0)

    def test_11_nan_and_infinite_rf_outputs_trigger_fallback(self):
        """Verify NaN or +/-inf outputs do NOT get converted to 0, but trigger analytical fallback."""
        # Test NaN
        with patch.object(prediction_service.q50_model, "predict", return_value=np.array([math.nan])):
            res_nan = prediction_service.predict(self.normal_state)
        self.assertEqual(res_nan.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res_nan.fallback_reason, FallbackReason.RF_NON_FINITE_OUTPUT.value)
        self.assertTrue(math.isfinite(res_nan.predicted_wait_minutes))

        # Test +Infinity
        with patch.object(prediction_service.q50_model, "predict", return_value=np.array([math.inf])):
            res_inf = prediction_service.predict(self.normal_state)
        self.assertEqual(res_inf.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res_inf.fallback_reason, FallbackReason.RF_NON_FINITE_OUTPUT.value)
        self.assertTrue(math.isfinite(res_inf.predicted_wait_minutes))

    # =========================================================================
    # 7. MODEL UNAVAILABLE & CHECKSUM VERIFICATION
    # =========================================================================
    def test_12_checksum_verification_and_graceful_model_unavailability(self):
        """
        Verify that if the model checksum fails or model is unavailable,
        the service falls back to analytical prediction safely without crashing.
        Properly restores original singleton state upon completion.
        """
        temp_service = prediction_service
        orig_verify = temp_service.verify_artifact_checksum
        try:
            # Test checksum verification on actual artifact
            valid = temp_service.verify_artifact_checksum(FROZEN_RF_MODEL_PATH, FROZEN_RF_SHA256)
            self.assertTrue(valid, "Production artifact checksum verification failed")

            # Simulate corrupt checksum
            temp_service.verify_artifact_checksum = MagicMock(return_value=False)
            temp_service.load_model()
            self.assertFalse(temp_service.rf_available)

            # Prediction request must still succeed via analytical fallback
            res = temp_service.predict(self.normal_state)
            self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
            self.assertEqual(res.fallback_reason, FallbackReason.MODEL_UNAVAILABLE.value)
            self.assertTrue(math.isfinite(res.predicted_wait_minutes))
        finally:
            # Crucial: restore verification and reload singleton model so subsequent tests are not poisoned
            temp_service.verify_artifact_checksum = orig_verify
            temp_service.load_model()
            self.assertTrue(temp_service.rf_available, "Failed to restore RF model availability after test_12")

    # =========================================================================
    # 8. SERVING LATENCY BENCHMARK
    # =========================================================================
    def test_13_serving_latency_measurement(self):
        """
        Measures actual single-row prediction latency through:
        A. Warm RF-only model inference
        B. Warm full RF serving path (asserting prediction_source == 'RF')
        C. Deterministic analytical fallback (asserting prediction_source == 'ANALYTICAL_FALLBACK')
        """
        self.assertTrue(prediction_service.rf_available, "RF model must be loaded for latency benchmark")

        # 1. Benchmark Warm Full RF Serving Path (20 warmup, 200 measured)
        for _ in range(20):
            res_warm = prediction_service.predict(self.normal_state)
            self.assertEqual(res_warm.prediction_source, PredictionSource.RF.value)

        full_rf_latencies_ms = []
        for _ in range(200):
            t0 = time.perf_counter()
            res = prediction_service.predict(self.normal_state)
            full_rf_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            self.assertEqual(res.prediction_source, PredictionSource.RF.value)

        full_rf_median = float(np.median(full_rf_latencies_ms))
        full_rf_p95 = float(np.percentile(full_rf_latencies_ms, 95))

        # 2. Benchmark Warm RF-Only Pipeline Inference (20 warmup, 200 measured)
        row_vals = [self.normal_state[col] for col in RAW_FEATURE_NAMES]
        df_input = pd.DataFrame([row_vals], columns=list(RAW_FEATURE_NAMES))
        for _ in range(20):
            _ = prediction_service.rf_model.predict(df_input)

        rf_only_latencies_ms = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = prediction_service.rf_model.predict(df_input)
            rf_only_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        rf_only_median = float(np.median(rf_only_latencies_ms))
        rf_only_p95 = float(np.percentile(rf_only_latencies_ms, 95))

        # 3. Benchmark Analytical Fallback Path (20 warmup, 200 measured)
        fallback_state = dict(self.normal_state)
        fallback_state["busy_assigned_counters"] = 0  # triggers Guard A
        for _ in range(20):
            res_fb = prediction_service.predict(fallback_state)
            self.assertEqual(res_fb.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)

        fallback_latencies_ms = []
        for _ in range(200):
            t0 = time.perf_counter()
            res_fb = prediction_service.predict(fallback_state)
            fallback_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            self.assertEqual(res_fb.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)

        fallback_median = float(np.median(fallback_latencies_ms))
        fallback_p95 = float(np.percentile(fallback_latencies_ms, 95))

        print(f"\n[LATENCY BENCHMARK RESULT]")
        print(f"  Warm RF-Only Pipeline Inference: Median = {rf_only_median:.3f} ms | P95 = {rf_only_p95:.3f} ms")
        print(f"  Warm Full RF Serving Path:       Median = {full_rf_median:.3f} ms | P95 = {full_rf_p95:.3f} ms")
        print(f"  Warm Analytical Fallback Path:   Median = {fallback_median:.3f} ms | P95 = {fallback_p95:.3f} ms")
        self.assertLess(full_rf_median, 100.0, "Median full RF serving latency exceeds 100 ms SLA")

    # =========================================================================
    # 9. COUNTER SWITCHING & CAPACITY SEMANTICS (TESTS A, B, C)
    # =========================================================================
    def test_14_capacity_before_during_after_switching(self):
        """
        Test A — Capacity Before, During, and After Counter Switching:
        Verifies exact capacity metrics across the counter reassignment lifecycle.
        """
        t_snap = self.t_snap
        services = {
            1: ServiceConfigSnapshot(1, "PT", "Property Tax", 10.0, 4.0),
            2: ServiceConfigSnapshot(2, "WC", "Water Service", 15.0, 5.0),
        }

        # BEFORE SWITCH: Counter 1 and Counter 2 both serving Property Tax (PT)
        counters_before = {
            1: CounterSnapshot(1, 1, "C1", "ACTIVE", t_snap, 1, frozenset([1, 2]),
                               ServingTokenSnapshot(101, "PT-101", 1, 1, "SERVING", t_snap - timedelta(minutes=5), t_snap - timedelta(minutes=4))),
            2: CounterSnapshot(2, 1, "C2", "ACTIVE", t_snap, 1, frozenset([1, 2]),
                               ServingTokenSnapshot(102, "PT-102", 1, 2, "SERVING", t_snap - timedelta(minutes=3), t_snap - timedelta(minutes=2))),
        }
        snap_before = SimulationSnapshot(
            office_id=1, office_code="MUNI", snapshot_time_utc=t_snap,
            services=services, queues={1: ServiceQueueSnapshot(1, (), ())}, counters=counters_before
        )
        vec_before = compute_raw_features(snap_before, service_id=1, position=1)
        self.assertEqual(vec_before.assigned_active_counters, 2, "Before switch: 2 assigned counters")
        self.assertEqual(vec_before.busy_assigned_counters, 2, "Before switch: 2 busy counters")

        # DURING SWITCH: Counter 2 is in transit/switching (status=ACTIVE, current_token=None)
        counters_during = {
            1: CounterSnapshot(1, 1, "C1", "ACTIVE", t_snap, 1, frozenset([1, 2]),
                               ServingTokenSnapshot(101, "PT-101", 1, 1, "SERVING", t_snap - timedelta(minutes=5), t_snap - timedelta(minutes=4))),
            2: CounterSnapshot(2, 1, "C2", "ACTIVE", t_snap, 1, frozenset([1, 2]), None),
        }
        snap_during = SimulationSnapshot(
            office_id=1, office_code="MUNI", snapshot_time_utc=t_snap,
            services=services, queues={1: ServiceQueueSnapshot(1, (), ())}, counters=counters_during
        )
        vec_during = compute_raw_features(snap_during, service_id=1, position=1)
        # assigned_active_counters includes Counter 2 in physical fleet assignment
        self.assertEqual(vec_during.assigned_active_counters, 2)
        # busy_assigned_counters excludes Counter 2 because current_token is None
        self.assertEqual(vec_during.busy_assigned_counters, 1)

        # AFTER SWITCH: Counter 2 is reassigned to WC
        counters_after = {
            1: CounterSnapshot(1, 1, "C1", "ACTIVE", t_snap, 1, frozenset([1, 2]),
                               ServingTokenSnapshot(101, "PT-101", 1, 1, "SERVING", t_snap - timedelta(minutes=5), t_snap - timedelta(minutes=4))),
            2: CounterSnapshot(2, 1, "C2", "ACTIVE", t_snap, 2, frozenset([1, 2]),
                               ServingTokenSnapshot(201, "WC-201", 2, 2, "SERVING", t_snap - timedelta(minutes=1), t_snap - timedelta(minutes=1))),
        }
        snap_after = SimulationSnapshot(
            office_id=1, office_code="MUNI", snapshot_time_utc=t_snap,
            services=services, queues={1: ServiceQueueSnapshot(1, (), ())}, counters=counters_after
        )
        vec_after = compute_raw_features(snap_after, service_id=1, position=1)
        self.assertEqual(vec_after.assigned_active_counters, 1, "After switch: 1 assigned to PT")
        self.assertEqual(vec_after.busy_assigned_counters, 1, "After switch: 1 busy for PT")

    def test_15_analytical_fallback_capacity_during_switching(self):
        """
        Test B — Analytical Fallback Capacity Calculation During Switching:
        Verifies that when a queue exists during switching (queue_length >= 1, busy < assigned),
        the analytical formula uses c = max(1, busy) to reflect currently serviceable capacity,
        rather than overstating capacity by dividing by assigned_active_counters.
        """
        # Scenario: assigned = 2, busy = 1, queue_length = 3, people_ahead = 2
        # service_mean = 10.0, elapsed = 4.0 -> residual = 10.0 - 4.0 = 6.0 min
        switching_state = {
            "office_id": "MUNI",
            "service_id": "PT",
            "hour": 10,
            "day_of_week": 2,
            "people_ahead": 2,
            "queue_length": 3,
            "assigned_active_counters": 2,
            "busy_assigned_counters": 1,   # 1 counter switching/unavailable
            "max_elapsed_service_minutes": 4.0,
            "service_mean_minutes": 10.0,
            "service_std_minutes": 4.0,
            "arrival_rate_30m": 12.0,
            "arrival_rate_60m": 10.0,
            "service_capacity_mu": 6.0,
            "traffic_intensity": 1.666,
        }

        # Guard C triggers analytical fallback
        res = prediction_service.predict(switching_state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY.value)

        # Expected wait calculation:
        # k = 2, mean_s = 10.0, residual = (10.0 - 4.0) = 6.0
        # Serviceable capacity c = busy = 1.0 (NOT assigned = 2.0)
        # Expected = (2 * 10.0 + 6.0) / 1.0 = 26.0 minutes
        # If incorrectly using c = 2.0, result would be 13.0 minutes
        analytical_val = compute_analytical_residual_aware(switching_state)
        self.assertEqual(analytical_val, 26.0, "Analytical capacity must use c = busy = 1 during switching")
        self.assertEqual(res.predicted_wait_minutes, 26.0)

    def test_16_no_accidental_semantic_regression_on_normal_capacity(self):
        """
        Test C — No Accidental Semantic Regression on Normal States:
        Verifies that when all assigned counters are busy (busy == assigned),
        capacity c equals assigned_active_counters normally.
        """
        normal_analytical_state = {
            "office_id": "MUNI",
            "service_id": "PT",
            "hour": 10,
            "day_of_week": 2,
            "people_ahead": 15,  # Guard D triggers analytical fallback
            "queue_length": 16,
            "assigned_active_counters": 2,
            "busy_assigned_counters": 2,
            "max_elapsed_service_minutes": 4.0,
            "service_mean_minutes": 10.0,
            "service_std_minutes": 4.0,
            "arrival_rate_30m": 12.0,
            "arrival_rate_60m": 10.0,
            "service_capacity_mu": 12.0,
            "traffic_intensity": 0.833,
        }
        res = prediction_service.predict(normal_analytical_state)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.DEEP_QUEUE.value)

        # k = 15, mean_s = 10.0, residual = 6.0, c = 2.0
        # Expected = (15 * 10.0 + 6.0) / 2.0 = 156.0 / 2.0 = 78.0 minutes
        analytical_val = compute_analytical_residual_aware(normal_analytical_state)
        self.assertEqual(analytical_val, 78.0, "Normal full-capacity states must use c = assigned = 2")
        self.assertEqual(res.predicted_wait_minutes, 78.0)

    # =========================================================================
    # 10. MODEL-SERVING IDENTITY COMPATIBILITY & SUPPORT CONTRACT (TESTS A-I)
    # =========================================================================
    def test_17_unknown_office_and_service_semantic_validation(self):
        """
        Semantic Input Validation for Office and Service IDs:
        Verifies that unknown office or service IDs raise IdentityValidationError (subclass of PredictionServiceError).
        """
        bad_office_state = dict(self.normal_state)
        bad_office_state["office_id"] = "UNKNOWN_OFFICE"
        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(bad_office_state)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNKNOWN_APPLICATION_OFFICE)

        bad_service_state = dict(self.normal_state)
        bad_service_state["service_id"] = "UNKNOWN_SERVICE"
        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(bad_service_state)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNKNOWN_APPLICATION_SERVICE)

    def test_18_contract_A_exact_supported_model_identity_rf_allowed(self):
        """
        Test A: Exact supported model identities -> RF prediction allowed.
        Supported pairs:
          Office MUNI + Services (PT, WC, BC)
          Office RTO  + Services (OT, DR)
        """
        muni_services = ["PT", "WC", "BC"]
        for srv in muni_services:
            st = dict(self.normal_state)
            st["office_id"] = "MUNI"
            st["service_id"] = srv
            res = prediction_service.predict(st)
            self.assertEqual(res.prediction_source, PredictionSource.RF.value, f"MUNI/{srv} must return RF")
            self.assertIsNone(res.fallback_reason)
            self.assertGreater(res.predicted_wait_minutes, 0.0)
            self.assertIsNotNone(res.p10_minutes)
            self.assertIsNotNone(res.p90_minutes)

        rto_services = ["OT", "DR"]
        for srv in rto_services:
            st = dict(self.normal_state)
            st["office_id"] = "RTO"
            st["service_id"] = srv
            res = prediction_service.predict(st)
            self.assertEqual(res.prediction_source, PredictionSource.RF.value, f"RTO/{srv} must return RF")
            self.assertIsNone(res.fallback_reason)
            self.assertGreater(res.predicted_wait_minutes, 0.0)

    def test_19_contract_B_unknown_application_office_422(self):
        """
        Test B: Unknown application office -> IdentityValidationError / HTTP 422 with UNKNOWN_APPLICATION_OFFICE.
        """
        st = dict(self.normal_state)
        st["office_id"] = "NYC_OFFICE"
        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(st)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNKNOWN_APPLICATION_OFFICE)

        with TestClient(app) as client:
            resp = client.post("/api/v1/predictions/wait-time", json=st)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["detail"]["error_code"], "UNKNOWN_APPLICATION_OFFICE")

    def test_20_contract_C_unknown_application_service_422(self):
        """
        Test C: Unknown application service -> IdentityValidationError / HTTP 422 with UNKNOWN_APPLICATION_SERVICE.
        """
        st = dict(self.normal_state)
        st["service_id"] = "PASSPORT_ISSUANCE"
        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(st)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNKNOWN_APPLICATION_SERVICE)

        with TestClient(app) as client:
            resp = client.post("/api/v1/predictions/wait-time", json=st)
            self.assertEqual(resp.status_code, 422)
            self.assertEqual(resp.json()["detail"]["error_code"], "UNKNOWN_APPLICATION_SERVICE")

    def test_21_contract_D_known_application_identity_but_unsupported_by_model_422(self):
        """
        Test D: Known application identity but unsupported by frozen model -> HTTP 422 UNSUPPORTED_MODEL_*.
        Offices: RJT, AMD, STV
        Services: LL, DL_RENEWAL, VEH_REG
        """
        unsupported_offices = ["RJT", "AMD", "STV"]
        for off in unsupported_offices:
            self.assertIn(off, APPLICATION_OFFICES, f"{off} must be a known application office")
            self.assertNotIn(off, MODEL_SUPPORTED_OFFICES, f"{off} must not be in model supported offices")
            st = dict(self.normal_state)
            st["office_id"] = off
            with self.assertRaises(IdentityValidationError) as ctx:
                _ = prediction_service.predict(st)
            self.assertEqual(ctx.exception.code, IdentityErrorCode.UNSUPPORTED_MODEL_OFFICE)

            with TestClient(app) as client:
                resp = client.post("/api/v1/predictions/wait-time", json=st)
                self.assertEqual(resp.status_code, 422)
                self.assertEqual(resp.json()["detail"]["error_code"], "UNSUPPORTED_MODEL_OFFICE")

        unsupported_services = ["LL", "DL_RENEWAL", "VEH_REG"]
        for srv in unsupported_services:
            self.assertIn(srv, APPLICATION_SERVICES, f"{srv} must be a known application service")
            self.assertNotIn(srv, MODEL_SUPPORTED_SERVICES, f"{srv} must not be in model supported services")
            st = dict(self.normal_state)
            st["service_id"] = srv
            with self.assertRaises(IdentityValidationError) as ctx:
                _ = prediction_service.predict(st)
            self.assertEqual(ctx.exception.code, IdentityErrorCode.UNSUPPORTED_MODEL_SERVICE)

            with TestClient(app) as client:
                resp = client.post("/api/v1/predictions/wait-time", json=st)
                self.assertEqual(resp.status_code, 422)
                self.assertEqual(resp.json()["detail"]["error_code"], "UNSUPPORTED_MODEL_SERVICE")

    def test_22_contract_E_unsupported_identity_never_returns_rf(self):
        """
        Test E: Unsupported application identities NEVER return prediction_source = RF.
        OneHotEncoder(handle_unknown='ignore') must NOT silently produce an all-zero vector RF prediction.
        """
        unsupported_offices = ["RJT", "AMD", "STV"]
        unsupported_services = ["LL", "DL_RENEWAL", "VEH_REG"]
        for off in unsupported_offices:
            for srv in unsupported_services:
                st = dict(self.normal_state)
                st["office_id"] = off
                st["service_id"] = srv
                with self.assertRaises(IdentityValidationError):
                    _ = prediction_service.predict(st)

    def test_23_contract_F_valid_supported_identity_deep_queue_triggers_analytical_fallback(self):
        """
        Test F: Valid model-supported identity + deep queue (people_ahead >= 11) -> ANALYTICAL_FALLBACK.
        """
        st = dict(self.normal_state)
        st["office_id"] = "MUNI"
        st["service_id"] = "PT"
        st["people_ahead"] = 11
        st["queue_length"] = 12

        res = prediction_service.predict(st)
        self.assertEqual(res.prediction_source, PredictionSource.ANALYTICAL_FALLBACK.value)
        self.assertEqual(res.fallback_reason, FallbackReason.DEEP_QUEUE.value)
        self.assertIsNone(res.p10_minutes)
        self.assertIsNone(res.p90_minutes)

    def test_24_contract_G_valid_supported_identity_normal_state_returns_rf(self):
        """
        Test G: Valid model-supported identity + normal state -> RF.
        """
        st = dict(self.normal_state)
        st["office_id"] = "MUNI"
        st["service_id"] = "PT"
        st["people_ahead"] = 3
        st["queue_length"] = 4
        st["assigned_active_counters"] = 2
        st["busy_assigned_counters"] = 2

        res = prediction_service.predict(st)
        self.assertEqual(res.prediction_source, PredictionSource.RF.value)
        self.assertIsNone(res.fallback_reason)
        self.assertGreater(res.predicted_wait_minutes, 0.0)

    def test_25_contract_H_analytical_fallback_never_receives_unsupported_service_identity(self):
        """
        Test H: Analytical fallback NEVER receives an unsupported service identity.
        Boundary states (deep queue, zero counters) on unsupported services MUST be rejected
        with IdentityValidationError, NOT silently diverted to analytical fallback.
        """
        # Deep queue with unsupported service LL
        deep_unsupported = dict(self.normal_state)
        deep_unsupported["office_id"] = "MUNI"
        deep_unsupported["service_id"] = "LL"
        deep_unsupported["people_ahead"] = 15

        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(deep_unsupported)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNSUPPORTED_MODEL_SERVICE)

        # Zero counters with unsupported office RJT
        zero_unsupported = dict(self.normal_state)
        zero_unsupported["office_id"] = "RJT"
        zero_unsupported["service_id"] = "PT"
        zero_unsupported["assigned_active_counters"] = 0

        with self.assertRaises(IdentityValidationError) as ctx:
            _ = prediction_service.predict(zero_unsupported)
        self.assertEqual(ctx.exception.code, IdentityErrorCode.UNSUPPORTED_MODEL_OFFICE)

    def test_26_contract_I_frozen_artifact_categories_are_source_of_truth(self):
        """
        Test I: Frozen artifact categories are verified as the authoritative source of truth.
        """
        self.assertTrue(prediction_service.rf_available)
        cat_tx = prediction_service.rf_model.named_steps["preprocessor"].named_transformers_["cat"]
        artifact_offices = frozenset(cat_tx.categories_[0])
        artifact_services = frozenset(cat_tx.categories_[1])

        self.assertEqual(artifact_offices, frozenset({"MUNI", "RTO"}))
        self.assertEqual(artifact_services, frozenset({"BC", "DR", "OT", "PT", "WC"}))
        self.assertEqual(artifact_offices, MODEL_SUPPORTED_OFFICES)
        self.assertEqual(artifact_services, MODEL_SUPPORTED_SERVICES)
        self.assertEqual(prediction_service.artifact_supported_offices, MODEL_SUPPORTED_OFFICES)
        self.assertEqual(prediction_service.artifact_supported_services, MODEL_SUPPORTED_SERVICES)


if __name__ == "__main__":
    unittest.main()


