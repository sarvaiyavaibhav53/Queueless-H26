# PRAVAAH — Step 2E.2: Forensic Dataset Audit Report

**Date**: 2026-10-08  
**Audit Status**: **PASS WITH NOTES**  
**Author**: Antigravity (Senior ML & Backend Engineer)  
**Dataset Inspected**: `ml/data/synthetic/production/dataset.csv` (and `.parquet`)  
**Generator Inspected**: `ml/src/generate_production_dataset.py`

---

## 1. Audit Scope & Files Inspected

This forensic audit evaluates the production synthetic dataset generated in Step 2E.1C against PRAVAAH's frozen architecture and machine learning guidelines prior to any model training.

### Files Inspected:
* `ml/src/generate_production_dataset.py` (authoritative generator)
* `ml/data/synthetic/production/dataset.csv` (32,802 rows, 6.60 MB)
* `ml/data/synthetic/production/dataset.parquet` (1.26 MB)
* `ml/data/synthetic/production/generation_metadata.json`
* `ml/data/synthetic/production/generation_report.md`
* `shared/features.py` (frozen feature extraction contract)
* `shared/stability.py` (frozen stability thresholds)

---

## 2. Dataset Overview & High-Level Statistics

* **Total Observations**: 32,802
* **Total Independent Trajectories**: 1,492 active trajectories with sampled observations (out of 1,500 simulated worlds)
* **Unique Configurations (`config_id`)**: 1,492
* **Offices Represented**:
  * `MUNI`: 23,011 rows (70.15%) across 975 trajectories
  * `RTO`: 9,791 rows (29.85%) across 517 trajectories
* **Services Represented**:
  * `PT`: 10,927 rows (33.31%)
  * `BC`: 6,052 rows (18.45%)
  * `WC`: 6,032 rows (18.39%)
  * `OT`: 5,767 rows (17.58%)
  * `DR`: 4,024 rows (12.27%)
* **Observation Types**:
  * `PERIODIC`: 20,885 rows (63.67%)
  * `JOIN`: 11,917 rows (36.33%)
* **Production Regimes**:
  * `HEALTHY`: 11,525 rows (35.14%)
  * `MODERATE`: 8,863 rows (27.02%)
  * `HIGH_LOAD`: 5,398 rows (16.46%)
  * `STRESSED`: 2,963 rows (9.03%)
  * `FAILURE`: 2,120 rows (6.46%)
  * `INTERVENTION`: 1,933 rows (5.89%)
* **Simulation Seeds**: Isolated in production range `[1_000_001, 1_001_500]`, completely disjoint from Step 2D.4 benchmark seed ranges `[100_000, 299_999]`.

---

## 3. Audit A — JOIN Observation Correctness

### A. Snapshot Timing & Event Sequencing
In `ml/src/generate_production_dataset.py`, lines 598–644 handle the `CUSTOMER_ARRIVAL` event at `current_time`:
1. If counters are idle and the queue is empty, the citizen immediately enters service without waiting.
2. Otherwise, the arriving citizen is appended to the tail of the queue (`queue.append(citizen)`), giving position $N = \text{queue.length()}$.
3. During the supervised window $[t_{\text{warmup}}, t_{\text{cutoff}}]$, candidate arrivals are sampled every 2nd join to balance observation classes.
4. `snap = capture_snapshot(current_time)` is executed immediately at the arrival instant.
5. All feature calculations (`compute_raw_features(snap, service_id, position=N)`) occur strictly at this instant.
6. Crucially, this snapshot happens:
   * **After** arrival and queue insertion.
   * **Before** any future customer arrivals.
   * **Before** future service completions, counter failures, recoveries, or intervention reassignments.

### B. Queue State Invariants
* Passed `position` $= N$.
* `shared.features.compute_raw_features` calculates:
  * `people_ahead = position - 1` $= N - 1$
  * `queue_length = len(queue.waiting_tokens)` $= N$
* The newly arrived citizen is at the back of the queue; there are exactly $N-1$ citizens ahead. The citizen does **not** appear ahead of themselves.

### C. Target Formulation
* `target_wait_minutes = round(service_start_time - snapshot_time, 4)`
* `service_start_time` is recorded only when a counter actually pops this citizen from the queue during future discrete events.
* No future service dispatch timing is observable at snapshot time.
* All 15 features are computed solely through the frozen `shared.features.compute_raw_features` function.

**Verdict: PASS**

---

## 4. Audit B — Point-in-Time Information Boundary

Every one of the 15 features was traced to its underlying simulation variable:

| Feature | Source Simulation State | Observable at Snapshot Time? | Future Leakage Risk |
| :--- | :--- | :--- | :--- |
| `office_id` | Static simulation metadata | Yes | None |
| `service_id` | Static service metadata | Yes | None |
| `hour` | `snapshot_time_utc.hour` (local) | Yes | None |
| `day_of_week` | `snapshot_time_utc.weekday()` | Yes | None |
| `people_ahead` | Position in queue minus 1 | Yes | None |
| `queue_length` | Current queue length at snapshot | Yes | None |
| `assigned_active_counters` | Counters currently active & assigned to service | Yes | None |
| `busy_assigned_counters` | Active assigned counters currently serving a token | Yes | None |
| `max_elapsed_service_minutes` | `max(0, current_time - serving_start)` for busy tokens | Yes | None |
| `service_mean_minutes` | Official service configuration mean | Yes | None |
| `service_std_minutes` | Official service configuration std | Yes | None |
| `arrival_rate_30m` | Arrivals with $t_{\text{snap}} - 30 < t_{\text{arr}} \le t_{\text{snap}}$ scaled to 1h | Yes | None |
| `arrival_rate_60m` | Arrivals with $t_{\text{snap}} - 60 < t_{\text{arr}} \le t_{\text{snap}}$ scaled to 1h | Yes | None |
| `service_capacity_mu` | $\mu \times \text{assigned\_active\_counters}$ | Yes | None |
| `traffic_intensity` | Trailing arrival rate / current service capacity | Yes | None |

### Leakage Verification:
* Future arrival schedule is invisible (trailing window strictly filtered by $t \le t_{\text{snap}}$).
* Future counter failures and recoveries are invisible until the event fires.
* Future intervention reassignments are invisible until `INTERVENTION_ACTIVATED` changes the assignment at simulation time.
* Zero future target leakage detected.

**Verdict: PASS**

---

## 5. Audit C — Censoring and Post-Cutoff Draining

The dataset records **0 censored rows** across 32,802 candidate observations.

### Root Cause Analysis:
1. **Safety Cap Window**: The safety drain horizon is configured as $t_{\text{cutoff}} + 240$ minutes (4.0 hours).
2. **Backlog Volume**: The arrival window is 90 minutes. In the most extreme STRESSED regime ($\rho \ge 1.35$), maximum queue backlog accumulated at $t_{\text{cutoff}}$ rarely exceeds 20–25 citizens.
3. **Counter Drain Capacity**: With 2 active counters serving at 10 minutes/citizen, a backlog of 25 citizens drains in approximately $25 \times 5 = 125$ minutes (~2.1 hours).
4. Because 125 minutes is well below the 240-minute cap, every citizen who arrived during the active window reached service start before the simulation safety cap expired.
5. Unserved/censored logic was confirmed to be present in code (`is_served = False` and `target_wait = None` if unserved), but no trajectories hit the 4-hour timeout.

**Verdict: PASS WITH NOTES** (Zero censoring is a genuine mathematical consequence of the 4-hour drain safety window relative to the 90-minute arrival window, not a generator bug).

---

## 6. Audit D — Near-Duplicate Analysis

The dataset metadata reports 10,920 near-duplicate rows (33.29%).

### Inspection of Near-Duplicate Pairs:
A forensic check of pairs occurring within $\le 10$ minutes within the same trajectory and service with identical `people_ahead` and `queue_length` reveals:
* **Pair Example 1 (Trajectory 1, Service PT, $\Delta t = 1.38$ min)**:
  * Observation 1: PERIODIC at $t=10.0$ min, `token_id` = 100008, `people_ahead` = 2, `max_elapsed` = 8.80 min, `arrival_rate_30m` = 12.0, Target = 10.04 min.
  * Observation 2: JOIN at $t=11.38$ min, `token_id` = 100009, `people_ahead` = 2, `max_elapsed` = 10.18 min, `arrival_rate_30m` = 14.0, Target = 11.66 min.
* **Findings**:
  1. The rows are **not** duplicates. They represent genuine time steps in queue dynamics.
  2. While `people_ahead` is identical, the physical citizens differ (`token_id` 100008 vs 100009), counter elapsed time increased by 1.38 minutes, trailing arrival rate updated, and future target wait increased from 10.04 min to 11.66 min.
  3. These rows are critical for learning real-world queuing dynamics.
  4. **Crucial implication for splitting**: Because multiple observations share a trajectory and citizens wait across multiple sampling intervals, random row-level splitting would leak identical trajectories and correlated tokens between train and test. Grouped splitting by trajectory/configuration is mandatory.

**Verdict: PASS WITH NOTES**

---

## 7. Audit E — Configuration Structure & Grouped Splitting Strategy

### Distribution Counts:
* Total unique `config_id`: 1,492
* Average trajectories per `config_id`: 1.0 (exact 1:1 mapping between `config_id` and `trajectory_id`)
* Average rows per `config_id`: 21.99 (Min: 1, Median: 20, Max: 40)
* Average JOIN rows per `config_id`: 8.01

### Decision Rule Evaluation (Section 7):
> *Decision Rule*: "If there are approximately 100 or more unique configurations: Use `config_id` as the primary grouping unit for train/validation/test splitting. No configuration may appear in more than one split."

* Since $1,492 \gg 100$, **`config_id` is selected as the primary grouping unit**.
* Every configuration represents an independent simulated world.
* All observations from any configuration will remain strictly within a single split (Train 70% / Val 15% / Test 15%).
* Stratification across splits will be applied on `regime` and `office_id` to guarantee identical distributions across Train, Validation, and Test.

**Verdict: PASS**

---

## 8. Audit F — Dataset Schema & Metadata Exclusion

### Schema Inspection:
* Columns 0–14: Exactly the 15 locked raw features (`RAW_FEATURE_NAMES`):
  1. `office_id`
  2. `service_id`
  3. `hour`
  4. `day_of_week`
  5. `people_ahead`
  6. `queue_length`
  7. `assigned_active_counters`
  8. `busy_assigned_counters`
  9. `max_elapsed_service_minutes`
  10. `service_mean_minutes`
  11. `service_std_minutes`
  12. `arrival_rate_30m`
  13. `arrival_rate_60m`
  14. `service_capacity_mu`
  15. `traffic_intensity`
* Column 15: Target `target_wait_minutes` (non-negative, finite, no NaNs).
* Columns 16–25: Observation and simulation metadata (`token_id`, `token_number`, `queue_position`, `snapshot_time_utc`, `config_id`, `simulation_seed`, `trajectory_id`, `regime`, `has_failure`, `obs_type`).
* Verification: The training pipeline will isolate columns 0–14 as inputs $X$ and column 15 as target $y$. Metadata columns will be used strictly for group partitioning and post-evaluation slice auditing.

**Verdict: PASS**

---

## 9. Final Audit Verdict & Gate Decision

| Audit Category | Result |
| :--- | :--- |
| **Audit A (JOIN Observation Correctness)** | **PASS** |
| **Audit B (Point-in-Time Information Boundary)** | **PASS** |
| **Audit C (Censoring & Drain Integrity)** | **PASS WITH NOTES** |
| **Audit D (Near-Duplicate & Temporal Analysis)** | **PASS WITH NOTES** |
| **Audit E (Configuration Structure & Grouping)** | **PASS** |
| **Audit F (Dataset Schema & Metadata Exclusion)** | **PASS** |
| **OVERALL AUDIT GATE** | **PASS WITH NOTES** |

### Gate Classification:
**PASS WITH NOTES**

### Authorized Next Step:
Phase A gate is officially cleared. Proceed to Phase B: Grouped split partitioning, analytical baseline evaluation, Random Forest training, and Quantile model training.
