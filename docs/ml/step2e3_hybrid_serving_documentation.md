# PRAVAAH — Step 2E.3: Frozen ML Hybrid Serving Integration Documentation

## 1. Overview & Purpose
This document details the deterministic hybrid serving boundary for PRAVAAH's wait-time prediction engine.
The production Random Forest model (`ml/models/production_wait_time_rf.joblib`, SHA-256: `99de21b0170fb856f4a8fdeaf70c8d1427c1e6739c9ad2a7e72c00c8a6a28af7`) is **frozen**. In accordance with the Step 2E.2A forensic audit, the RF model possesses high accuracy within its training distribution but encounters physical degradation in specific boundary conditions (e.g., initial dispatch states where counters have not yet become busy, partially degraded capacity, or deep queues).

To guarantee strict safety, monotonic prediction intervals, and deterministic fallbacks, PRAVAAH implements a **pre-inference hybrid serving policy**:
```text
Simulation Snapshot / Backend Request
                 ↓
     Shared Feature Construction
        (15 Canonical Features)
                 ↓
    Hybrid Serving Policy Evaluation
       (Guards A, B, C, and D)
        /                     \
       /                       \
[All Guards Pass]      [Any Guard Triggers OR Model Unavailable]
      ↓                                    ↓
Frozen RF Pipeline               Deterministic Residual-Aware
 (+ Empirical P10/P90)                 Analytical Method
      ↓                                    ↓
Monotonic Quantile Check             Single Point Estimate
 & Finite Range Clipping              (Null Quantile Bounds)
       \                                   /
        \                                 /
         Canonical Prediction Response Object
```

---

## 2. Serving-Safety Guards (Pre-Inference Evaluation)

The hybrid serving boundary evaluates the following deterministic safety guards **strictly before invoking the Random Forest model**:

### Guard A — Waiting Queue with Zero Busy Counters
* **Condition:** `queue_length >= 1` AND `busy_assigned_counters == 0`
* **Serving Action:** Residual-aware analytical fallback.
* **Fallback Reason:** `QUEUE_WITH_ZERO_BUSY_COUNTERS`
* **Rationale:** As proven in the Step 2E.2A forensic audit, the production training dataset (1,500 intended trajectories, 1,492 retained/non-empty trajectories) contains zero observations with a waiting queue and zero busy assigned counters. In initial dispatch scenarios (such as system startup or preloaded queues), the RF model incorrectly predicts non-zero delays (6.5–11.3 minutes) for the citizen at the front of the queue who will be served immediately ($t=0$). The analytical residual-aware estimator correctly returns 0.0.

### Guard B — Zero Active Assigned Counters
* **Condition:** `assigned_active_counters == 0`
* **Serving Action:** Residual-aware analytical fallback.
* **Fallback Reason:** `ZERO_ACTIVE_COUNTERS`
* **Rationale:** Total service shutdown or unavailability. The analytical model safely handles zero counters via protected denominator scaling.

### Guard C — Waiting Queue with Partially Unavailable Assigned Capacity
* **Condition:** `queue_length >= 1` AND `busy_assigned_counters < assigned_active_counters`
* **Serving Action:** Residual-aware analytical fallback.
* **Fallback Reason:** `PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY`
* **Rationale:** Counters undergoing reassignment delay or dynamic switching are not yet actively serving. The analytical method directly factors instantaneous available capacity.

### Guard D — Deep Queue
* **Condition:** `people_ahead >= DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD` (where `DEEP_QUEUE_FALLBACK_PEOPLE_AHEAD = 11`)
* **Serving Action:** Residual-aware analytical fallback.
* **Fallback Reason:** `DEEP_QUEUE`
* **Audited Empirical Evidence (from `dataset.parquet`, Stratified Grouped Split Seed 42):**
  * **VAL (all rows, `people_ahead >= 11`, 69 rows, 12 trajectories):**
    * RF: MAE = 7.688 min, RMSE = 9.305 min
    * Residual Analytical: MAE = 5.059 min, RMSE = 6.151 min
    * Analytical Advantage: **+2.629 min MAE** (+3.154 min RMSE)
  * **VAL (JOIN only, `people_ahead >= 11`, 43 rows, 11 trajectories):**
    * RF: MAE = 7.490 min, RMSE = 9.222 min
    * Residual Analytical: MAE = 4.993 min, RMSE = 6.220 min
    * Analytical Advantage: **+2.497 min MAE** (+3.001 min RMSE)
  * **TEST (all rows, `people_ahead >= 11`, 48 rows, 9 trajectories):**
    * RF: MAE = 6.298 min, RMSE = 7.898 min
    * Residual Analytical: MAE = 5.320 min, RMSE = 6.535 min
    * Analytical Advantage: **+0.978 min MAE** (+1.363 min RMSE)
  * **TEST (JOIN only, `people_ahead >= 11`, 32 rows, 9 trajectories):**
    * RF: MAE = 6.252 min, RMSE = 7.978 min
    * Residual Analytical: MAE = 5.034 min, RMSE = 6.192 min
    * Analytical Advantage: **+1.218 min MAE** (+1.787 min RMSE)

  *(Correction Note: The previous draft report mentioned "Analytical MAE 1.84 min vs RF MAE 3.21 min", which was an unverified conflation. The true audited empirical metrics above confirm that the deterministic residual-aware analytical method decisively outperforms the frozen RF model across both validation and test sets for `people_ahead >= 11`.)*

---

## 3. Identity Taxonomy, Support Contract & Compatibility Policy

### Identity Space Definition
1. **Application Identity Taxonomy (Database & Backend APIs):**
   * **Offices:** `{"MUNI", "RTO", "RJT", "AMD", "STV"}`
     * `MUNI`: Municipality (discrete-event simulation scenarios & production dataset)
     * `RTO`: Regional Transport Office (discrete-event simulation scenarios & production dataset)
     * `RJT`: Rajkot RTO (relational database models, backend queue tests, legacy dataset)
     * `AMD`: Ahmedabad Office (relational database models, backend queue tests, legacy dataset)
     * `STV`: Surat Office (relational database models, legacy dataset)
   * **Services:** `{"PT", "WC", "BC", "OT", "DR", "LL", "DL_RENEWAL", "VEH_REG"}`
     * `PT`: Property Tax (simulation scenarios, production dataset)
     * `WC`: Water Service (simulation scenarios, production dataset)
     * `BC`: Birth Certificate (simulation scenarios, production dataset)
     * `OT`: Vehicle Ownership Transfer (simulation scenarios, production dataset)
     * `DR`: Duplicate/Replacement RC (simulation scenarios, production dataset)
     * `LL`: Learner License (backend queue tests, legacy dataset)
     * `DL_RENEWAL`: Driving License Renewal (backend queue tests, legacy dataset)
     * `VEH_REG`: Vehicle Registration (backend queue tests, legacy dataset)

2. **Frozen Model Supported Identity Space (`production_wait_time_rf.joblib` v2.0):**
   * **Offices:** `MODEL_SUPPORTED_OFFICES = frozenset({"MUNI", "RTO"})`
   * **Services:** `MODEL_SUPPORTED_SERVICES = frozenset({"BC", "DR", "OT", "PT", "WC"})`
   * Preprocessor: `OneHotEncoder(categories=[['MUNI', 'RTO'], ['BC', 'DR', 'OT', 'PT', 'WC']], handle_unknown='ignore')`

3. **Compatibility Matrix:**
   | Identity Code | Type | Application Known? | Frozen Model Supported? | Serving Policy Behavior |
   | :--- | :--- | :--- | :--- | :--- |
   | `MUNI` | Office | Yes | **Yes** | Direct RF evaluation allowed |
   | `RTO` | Office | Yes | **Yes** | Direct RF evaluation allowed |
   | `RJT` | Office | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_OFFICE` (Retraining required) |
   | `AMD` | Office | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_OFFICE` (Retraining required) |
   | `STV` | Office | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_OFFICE` (Retraining required) |
   | `PT` | Service | Yes | **Yes** | Direct RF evaluation allowed (mean=10.0m, std=4.0m) |
   | `WC` | Service | Yes | **Yes** | Direct RF evaluation allowed (mean=15.0m, std=6.0m) |
   | `BC` | Service | Yes | **Yes** | Direct RF evaluation allowed (mean=7.0m, std=2.8m) |
   | `OT` | Service | Yes | **Yes** | Direct RF evaluation allowed (mean=14.0m, std=5.6m) |
   | `DR` | Service | Yes | **Yes** | Direct RF evaluation allowed (mean=8.0m, std=3.2m) |
   | `LL` | Service | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_SERVICE` (Retraining required) |
   | `DL_RENEWAL` | Service | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_SERVICE` (Retraining required) |
   | `VEH_REG` | Service | Yes | **No** | **HTTP 422** `UNSUPPORTED_MODEL_SERVICE` (Retraining required) |
   | `*` (any other) | Office | No | No | **HTTP 422** `UNKNOWN_APPLICATION_OFFICE` |
   | `*` (any other) | Service | No | No | **HTTP 422** `UNKNOWN_APPLICATION_SERVICE` |

### Two-Tier Identity Validation Contract
1. **Tier 1 (Application Taxonomy Check):** If `office_id` or `service_id` is unrecognized by the backend application taxonomy, reject with `IdentityErrorCode.UNKNOWN_APPLICATION_OFFICE` or `IdentityErrorCode.UNKNOWN_APPLICATION_SERVICE` (HTTP 422).
2. **Tier 2 (Frozen Model Contract Check):** If `office_id` or `service_id` is a valid application identity but was not present in the frozen model's training distribution (`{"RJT", "AMD", "STV"}` or `{"LL", "DL_RENEWAL", "VEH_REG"}`), reject with `IdentityErrorCode.UNSUPPORTED_MODEL_OFFICE` or `IdentityErrorCode.UNSUPPORTED_MODEL_SERVICE` (HTTP 422).
3. **No Silent `handle_unknown="ignore"` Coercion:** The serving boundary must **never** send an unseen categorical value through `OneHotEncoder(handle_unknown="ignore")` to generate an all-zero vector RF prediction.
4. **No Analytical Fallback Masking:** Unsupported categorical identities are strictly prohibited from entering analytical fallback. Analytical fallback is reserved solely for model-supported identities encountering boundary states.
5. **No Invented Mappings:** The serving boundary does not alias or map `RJT -> RTO` or `LL -> PT`. Unsupported identities are rejected explicitly until a new model version is trained.

---

## 4. Canonical Prediction Contract & Closed Enums

### Prediction Source Closed Enum (`PredictionSource`)
* `RF`: Prediction generated via frozen Random Forest model and quantile regressors.
* `ANALYTICAL_FALLBACK`: Prediction generated via deterministic residual-aware analytical queue formula.

### Fallback Reason Closed Enum (`FallbackReason`)
* `ZERO_ACTIVE_COUNTERS`: No counters assigned to service.
* `QUEUE_WITH_ZERO_BUSY_COUNTERS`: Queue exists but no assigned counter has begun serving.
* `PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY`: Queue exists and some assigned counters are idle/switching (`busy < assigned`).
* `DEEP_QUEUE`: `people_ahead >= 11`, beyond reliable RF support.
* `RF_NON_FINITE_OUTPUT`: Model output returned NaN or $\pm\infty$.
* `MODEL_UNAVAILABLE`: Artifact missing, corrupted, or failed checksum verification.

### Identity Error Code Closed Enum (`IdentityErrorCode`)
* `UNKNOWN_APPLICATION_OFFICE`: Office ID unrecognized by application taxonomy.
* `UNKNOWN_APPLICATION_SERVICE`: Service ID unrecognized by application taxonomy.
* `UNSUPPORTED_MODEL_OFFICE`: Known application office, but unsupported by frozen model artifact v2.0.
* `UNSUPPORTED_MODEL_SERVICE`: Known application service, but unsupported by frozen model artifact v2.0.

### RF Prediction Contract
When all guards pass, identity is supported, and the frozen RF model is available:
* `predicted_wait_minutes`: P50 point prediction from the pipeline.
* `p10_minutes`: Empirical P10 prediction quantile from auxiliary model.
* `p90_minutes`: Empirical P90 prediction quantile from auxiliary model.
* `prediction_source`: `"RF"`
* `fallback_reason`: `None` (`null`)

### Analytical Fallback Contract
When any guard triggers or the RF model is unavailable for a model-supported identity:
* `predicted_wait_minutes`: Deterministic residual-aware estimate `(k * mean_s + residual) / c`.
* `p10_minutes`: `None` (`null`)
* `p90_minutes`: `None` (`null`)
* `prediction_source`: `"ANALYTICAL_FALLBACK"`
* `fallback_reason`: Machine-readable closed enum string.

---

## 5. Anomaly Handling & Monotonic Quantile Post-Processing

1. **Non-Finite Model Outputs (NaN, +inf, -inf):**
   * Detected immediately. Never clipped to zero.
   * Model invocation triggers analytical fallback with `fallback_reason="RF_NON_FINITE_OUTPUT"`.
2. **Negative Predictions:**
   * Predictions $< 0.0$ are clipped to `0.0` and logged as anomalies.
3. **Quantile Crossing ($P10 > P50$ or $P50 > P90$):**
   * Checked deterministically.
   * Monotonic correction policy:
     $$P50_{\text{adj}} = \max(P10, P50)$$
     $$P90_{\text{adj}} = \max(P50, P90)$$
   * Anomaly logged; client receives strictly ordered quantiles $P10 \le P50 \le P90$.

---

## 6. Model Loading Lifecycle & Checksum Verification

* **Singleton Lifecycle:** `WaitTimePredictionService` is implemented with a thread-safe singleton pattern (`threading.Lock()`).
* **Artifact Path:** `ml/models/production_wait_time_rf.joblib` (8.75 MB).
* **SHA-256 Checksum:** `99de21b0170fb856f4a8fdeaf70c8d1427c1e6739c9ad2a7e72c00c8a6a28af7`.
* **Model Type:** Scikit-Learn `Pipeline` (`ColumnTransformer` + `RandomForestRegressor`, 100 estimators, `max_depth=16`, `min_samples_leaf=5`, `n_jobs=1`).
* **Runtime Versions:** Scikit-Learn `1.8.0`, Joblib `1.5.3`.
* **Category Verification on Load:** During startup, `load_model()` asserts that the joblib artifact's categorical transformer categories exactly match `MODEL_SUPPORTED_OFFICES` and `MODEL_SUPPORTED_SERVICES`.
* **Integrity Enforcement:** Verified at startup/first load. If corrupted or missing, `rf_available` is set to `False`, and all supported-identity prediction requests seamlessly degrade to analytical fallback without application crashes.

---

## 7. Latency Forensic Audit & Benchmark Results

### Latency Audit Results (Forensic Verification)
Benchmarked with 20 warmup calls excluded and 200 measured repetitions:
* **Warm RF-Only Pipeline Inference:** Median `19.701 ms` | P95 `31.646 ms`
* **Warm Full RF Serving Path:** Median `54.704 ms` | P95 `65.561 ms` (includes two-tier identity validation, 15-feature DataFrame construction, serving guards, base RF pipeline, and 3 quantile regressors)
* **Warm Analytical Fallback Path:** Median `0.014 ms` | P95 `0.018 ms`
* **Verification:** `prediction_source == "RF"` verified on 100% of RF benchmark calls; `prediction_source == "ANALYTICAL_FALLBACK"` verified on 100% of fallback calls.

---

## 8. Counter Switching Capacity Semantics

* **Canonical Feature Meaning:** `assigned_active_counters` represents physical assignment fleet size (`status == "ACTIVE"` and `assigned_service_id == service_id`). Counters in transit during switching maintain `status == "ACTIVE"` and are included in `assigned_active_counters`.
* **Busy Metrics:** `busy_assigned_counters` strictly requires `current_token is not None`. Switching counters have `current_token is None` and are excluded from `busy_assigned_counters`.
* **Guard Trigger:** During switching with a waiting queue (`queue_length >= 1`), `busy < assigned`. **Guard C triggers immediate analytical fallback (`PARTIALLY_UNAVAILABLE_ASSIGNED_CAPACITY`)**.
* **Capacity $c$ Correction:** In `compute_analytical_residual_aware`, when `queue_length >= 1` and `busy >= 1` and `busy < assigned`, capacity is set to $c = \max(1, busy)$. This prevents overstating capacity by dividing by transitioning counters, guaranteeing wait times are computed strictly from operationally serviceable capacity.

---

## 9. Architecture Scope & Production Citizen ETA Reality

* **Feature-Level Prediction Endpoint (`/api/v1/predictions/wait-time`):**
  * Direct endpoint accepting 15 features in request body.
  * Strictly enforces two-tier identity validation (`UNKNOWN_APPLICATION_*` vs `UNSUPPORTED_MODEL_*` $\to$ HTTP 422).
  * Evaluates authoritative frozen model or analytical fallback for supported identities.
* **Production Citizen/Employee ETA Path:**
  * End-to-end production flow: `authenticated request -> office/service/token -> authoritative DB queue state -> snapshot -> feature construction -> prediction service -> response`.
  * **Status:** `NOT IMPLEMENTED`.
  * The production database queue flow currently uses mock arrival rates and does not construct a live `SimulationSnapshot` via `shared/features.py`.

---

## 10. Test Execution & Database Compatibility

* **Test Suites Executed:**
  * `tests/test_ml_hybrid_serving.py` (26 tests passed)
  * `backend/tests/test_api.py` (10 tests passed)
  * `backend/tests/test_phase3_ml_integration.py` (5 tests passed)
  * `tests/test_ml_production_model.py` (11 tests passed)
  * `tests/test_ml_production_dataset.py` (12 tests passed)
  * `tests/test_ml_dataset_proof.py` (14 tests passed)
  * `tests/test_ml_sampling_benchmark.py` (8 tests passed)
  * `tests/test_shared_features.py` (26 tests passed)
  * `tests/test_intervention_evaluator.py` (18 tests passed)
  * `tests/test_scenario_benchmarks.py` (32 tests passed)
  * Full test discovery `tests/test_*.py`: 229 tests passed, 0 failed.
* **Database Driver Environment:**
  * All database-dependent integration tests were executed against **SQLite** (`$env:DATABASE_URL = 'sqlite:///./test.db'`).
  * **PostgreSQL integration:** `NOT RUN` (psycopg driver is not installed in the global evaluation environment; SQLite tests do not prove PostgreSQL compatibility).

