# PRAVAAH — Forensic Codebase Audit Report

**Project:** PRAVAAH (formerly QueueLess Government Office)  
**Location:** `c:\Users\vaibh\OneDrive\Desktop\h-26\pravaah`  
**Role:** Senior Backend + ML Engineer  
**Audit Scope:** Full Forensic Read-Only Audit (Backend, ML Simulation, Data Pipelines, Models, APIs, DB Schema)  
**Date:** 2026-10-07  

---

## 1. Executive Summary

A comprehensive, read-only forensic inspection was conducted across the existing PRAVAAH repository to evaluate its alignment with the target PRAVAAH architecture (individual eligible counters, per-service queues, queue stability modeling, simulation-based intervention optimization, and shared feature contracts).

### Core Findings:
1. **Working Baseline:** The codebase contains a functional, well-tested FastAPI backend with PostgreSQL persistence, JWT authentication (bcrypt + HS256), role enforcement (`citizen`, `employee`, `admin`), row-level concurrency locking (`with_for_update()`), WebSockets, and a scikit-learn Random Forest model pipeline (~723 MB joblib) serving wait-time point predictions.
2. **Fundamental Simulator Incompatibility:** The current simulator (`ml/src/queue_engine.py` and `ml/src/counter_manager.py`) simulates **one single office-wide FIFO queue** across all services and tracks counters merely as a **scalar integer tally** (`counters_free: int`). It does **not** model individual counter entities, service eligibility, per-service queues, counter assignments, switching delays, or state snapshots.
3. **Severe Feature Inconsistencies & Synthetic Collinearity:** The trained Random Forest model was fitted on data generated from this single-queue simulator where `queue_length` and `people_ahead` are 100% duplicate features ($r = 1.0$), and `active_counters` was fixed to static office capacity. In the backend (`backend/app/api/queues.py:91-105`), `arrival_rate_15m` is hard-coded to `5` and `no_show_rate` is hard-coded to `0.05`. There is **no shared feature contract**.
4. **Intervention Engine Non-Existent:** The repository currently lacks the PRAVAAH intelligence core: queue stability ($\lambda, \mu$), time-to-congestion, candidate intervention simulations (e.g., 15/30/60 min reassignments), same-seed baseline comparisons (Naive longest-queue vs. Optimizer), and uncertainty bounds.
5. **Security Gaps:** The backend lacks CORS middleware (blocking React frontends), uses a fallback hard-coded JWT secret (`"super-secret-key-for-development"`), leaves `/api/v1/predictions/wait-time` completely unauthenticated, and grants employees global unscoped access across all offices.

---

## 2. Current Architecture

The existing system operates as a standard monolithic CRUD + ML inference loop:

```
[Citizen Client]
     │
     ├──> POST /api/v1/auth/login ──> Issues JWT
     │
     ├──> POST /api/v1/queues/{service_id}/join
     │         │
     │         ├──> queue_service.generate_token()
     │         │       └──> Increments daily sequence in `queue_counters` (Row Lock)
     │         │       └──> Inserts row into `queue_tokens` (status: WAITING)
     │         │
     │         ├──> Ad-hoc Feature Dict Assembly (backend/app/api/queues.py:91-105)
     │         │       ├──> Hard-coded: arrival_rate_15m = 5
     │         │       ├──> Hard-coded: no_show_rate = 0.05
     │         │       └──> Static: active_counters = office.counters
     │         │
     │         ├──> prediction_service.predict_wait_time()
     │         │       └──> Pipeline.predict() using 723MB joblib (Random Forest)
     │         │       └──> Saves single integer: token.predicted_wait_minutes
     │         │
     │         └──> Background Task: ws_manager.broadcast_queue_update("JOINED")
     │
     └──> WS /ws/queues/{office_id}/{service_id}?token=... ──> Receives queue updates

[Employee Client]
     │
     ├──> POST /api/v1/queues/{office_id}/{service_id}/call-next ──> WAITING -> CALLED
     ├──> POST /api/v1/tokens/{token_id}/start                  ──> CALLED -> SERVING
     ├──> POST /api/v1/tokens/{token_id}/complete               ──> SERVING -> COMPLETED
     └──> POST /api/v1/tokens/{token_id}/skip                   ──> CALLED -> SKIPPED
          (All trigger ws_manager.broadcast_queue_update; NO counter entity updated)
```

---

## 3. Repository / File Map

| Path | Area | Primary Responsibility |
|---|---|---|
| `backend/app/main.py` | Backend | FastAPI application root, lifespan startup model loading, router inclusion. |
| `backend/app/core/config.py` | Backend Core | App configuration, database URL, JWT secret and token expiration settings. |
| `backend/app/core/security.py` | Backend Core | Bcrypt hashing/verification, HS256 JWT encoding. |
| `backend/app/db/database.py` | Backend DB | SQLAlchemy engine, session maker (`SessionLocal`), and `get_db` dependency. |
| `backend/app/models/office.py` | Domain Models | `Office` entity (`id`, `code`, `name`, `counters: int`, `demand_multiplier`). |
| `backend/app/models/service.py` | Domain Models | `Service` entity (`id`, `code`, `name`, `mean_minutes`, `std_minutes`). |
| `backend/app/models/queue_counter.py` | Domain Models | **Misnomer**: Daily ticket sequence tracker (`office_id`, `service_id`, `date`, `current_number`). |
| `backend/app/models/token.py` | Domain Models | `QueueToken` entity (`token_number`, status enum, timestamps, `predicted_wait_minutes`). |
| `backend/app/models/user.py` | Domain Models | `User` entity (`email`, `password_hash`, `role` enum, `is_active`). |
| `backend/app/api/auth.py` | API | Registration, login, current user endpoint (`/me`). |
| `backend/app/api/queues.py` | API | Citizen `join_queue`, staff `call_next_token`, queue status and active tokens views. |
| `backend/app/api/tokens.py` | API | Token state mutations (`start`, `complete`, `skip`, `cancel`), queue position query. |
| `backend/app/api/predictions.py` | API | Direct wait-time prediction endpoint (`/api/v1/predictions/wait-time`). |
| `backend/app/api/analytics.py` | API | Admin historical analytics across offices, services, queues, and ML performance. |
| `backend/app/api/ws.py` | Real-time | WebSocket endpoint handler with query/header JWT authentication. |
| `backend/app/ws_manager.py` | Real-time | In-memory connection pool and queue update broadcast dispatcher. |
| `backend/app/services/queue_service.py` | Business Logic | Token generation with row lock, queue position calculation, state transitions. |
| `backend/app/services/prediction_service.py` | Business Logic | Singleton wrapper loading scikit-learn pipeline and invoking `.predict()`. |
| `backend/alembic/versions/` | Migrations | 3 Alembic migration scripts defining the database schema. |
| `ml/config/simulation_config.yaml` | ML Config | Config for offices, services, arrival rates, service times, and downtime. |
| `ml/src/queue_engine.py` | Simulator | Simple FIFO queue list per office (`QueueEngine`). |
| `ml/src/counter_manager.py` | Simulator | Integer decrement/increment counter allocator per office (`CounterManager`). |
| `ml/src/arrival_process.py` | Simulator | Multinomial hourly arrival generator (`ArrivalProcess`). |
| `ml/src/service_process.py` | Simulator | Lognormal service time duration sampler (`ServiceProcess`). |
| `ml/src/event_engine.py` | Simulator | Heapq discrete-event simulation engine (`EventEngine`). |
| `ml/src/simulate.py` | Simulator | Entrypoint running test simulation (5 days). |
| `ml/src/generate_full.py` | Dataset Gen | Entrypoint generating full 365-day dataset (~195k events). |
| `ml/feature_engineering.py` | Feature Eng | Offline dataset script producing modeling parquet files. |
| `ml/src/baseline.py` | ML Baseline | Chronological dataset splitter and formula-based baseline evaluator. |
| `ml/src/train_random_forest.py` | ML Training | Random Forest Regressor hyperparameter tuning, training, and export. |
| `ml/models/random_forest_wait_time_pipeline.joblib` | Artifacts | 723 MB fitted Scikit-Learn Pipeline artifact. |
| `ml/models/random_forest_metadata.json` | Artifacts | Model metadata, evaluation scores, and hyperparameters. |

---

## 4. Simulator Audit

### Detailed Architecture Analysis:
- **Event Engine:** Implemented in `ml/src/event_engine.py:9-130` as `class EventEngine`. Uses Python's `heapq` storing tuples `(timestamp, event_counter, event_type, data)`. Supports 3 event types: `'ARRIVAL'`, `'SERVICE_END'`, and `'COUNTER_BACK'`.
- **Counter Manager:** Implemented in `ml/src/counter_manager.py:1-20` as `class CounterManager`. It holds a dictionary `self.counters_free = {o: config['offices'][o]['counters']}`. When allocated, it decrements the integer; when released, it increments the integer. **Counters have no identities, eligibility, or assigned services.**
- **Arrival Generation:** Implemented in `ml/src/arrival_process.py:6-45` as `class ArrivalProcess`. Uses `np.random.normal` for daily volume variation and `np.random.multinomial` for hourly volume distribution based on weights defined in `ml/config/simulation_config.yaml:37-45`.
- **Queue Implementation:** Implemented in `ml/src/queue_engine.py:1-13` as `class QueueEngine`. Uses `self.queues = {o: [] for o in offices}`. Each office has a single FIFO Python list where citizens of all services are merged.
- **Service Completion:** In `ml/src/event_engine.py:96-112` `handle_service_end()`. Sets `actual_service_minutes = (ts - person['service_start_time']).total_seconds() / 60.0` and appends to `self.completed_events`.
- **Service-Time Generation:** Implemented in `ml/src/service_process.py:4-22` as `class ServiceProcess`. Samples from lognormal distribution matching service `mean_minutes` and `std_minutes`, with a 3% probability of being multiplied by 3.0 (`long_service`).
- **Counter Failure/Recovery:** Rudimentarily simulated in `ml/src/event_engine.py:104-108`. After service end, with probability 0.02, counter release is deferred by scheduling a `COUNTER_BACK` event after a random uniform delay (5–15 mins). No specific counter is tracked as down.
- **Reassignment / Switching Delay:** **Not implemented.**
- **Cancellation / No-Show:** In `ml/src/event_engine.py:60-91`. Evaluated **immediately at arrival timestamp**. If flagged as cancelled or no-show, the citizen is dropped immediately and never enqueued.
- **Randomness & Seeding:** Handled only in top-level scripts (`ml/src/simulate.py:17-18` and `ml/src/generate_full.py:19-20`) calling `np.random.seed(42)` and `random.seed(42)`. Neither `EventEngine` nor its components accept an independent RNG seed or state instance.
- **Target Wait Calculation:** In `ml/src/event_engine.py:124` inside `try_assign_counter`: `actual_wait_minutes = (ts - person['arrival_time']).total_seconds() / 60.0`.

---

### Verification of the 18 Specific Questions:

| # | Question | Verification | Exact Repository Citation |
|---|---|---|---|
| 1 | Is there one queue per service? | **NO** | `ml/src/queue_engine.py:3`: `self.queues = {o: [] for o in offices}`. All services in an office share a single FIFO queue list. |
| 2 | Are counters individual entities? | **NO** | `ml/src/counter_manager.py:3`: `self.counters_free = {o: config['offices'][o]['counters'] for o in config['offices']}`. Counters are only an integer count. |
| 3 | Does each counter have eligibility? | **NO** | `ml/src/event_engine.py:119`: `try_assign_counter` checks only if `counters_free > 0`; any free counter takes the first citizen regardless of service. |
| 4 | Can counters have assignments? | **NO** | `ml/src/counter_manager.py:12-16`: `allocate()` only decrements an integer counter. |
| 5 | Can counters fail/recover? | **PARTIALLY** | `ml/src/event_engine.py:104-108`: Counter downtime delays `release()` via a `COUNTER_BACK` event, but no individual counter entity is tracked. |
| 6 | Can counters be reassigned? | **NO** | Neither `ml/src/counter_manager.py` nor `ml/src/event_engine.py` contains reassignment logic. |
| 7 | Is reassignment time-based? | **NO** | Reassignment does not exist in the codebase. |
| 8 | Is switching delay represented? | **NO** | Switching delay does not exist in the codebase. |
| 9 | Can simulation start from an arbitrary state snapshot? | **NO** | `ml/src/event_engine.py:25-30`: `run_day()` hard-resets all queues and counters to empty/fresh initial state. |
| 10 | Is randomness seedable? | **PARTIALLY (Script-level only)** | `ml/src/generate_full.py:19-20`: Seeds are set globally via `np.random.seed()` and `random.seed()`. Simulator classes do not manage local RNG instances. |
| 11 | Can identical seeds reproduce identical scenarios? | **YES (Global full runs only)** | Verified in `ml/models/RANDOM_FOREST_REPORT.md:259`. However, sub-scenarios/replications cannot be independently seeded without resetting global state. |
| 12 | Is target waiting time generated from actual queue behavior? | **YES** | `ml/src/event_engine.py:124`: `actual_wait_minutes = (ts - person['arrival_time']).total_seconds() / 60.0` when dequeued by an available counter. |
| 13 | Are arrivals time-varying? | **YES** | `ml/src/arrival_process.py:14-28`: Uses hourly weights from config to distribute daily citizens over operating hours. |
| 14 | Are burst events supported? | **NO** | `ml/src/arrival_process.py:28-34`: Arrivals are uniformly distributed at random minutes/seconds within each hour slot. No surge or burst logic exists. |
| 15 | Are service times realistically distributed? | **YES** | `ml/src/service_process.py:8-21`: Follows lognormal distribution with long-service heavy-tail probability (0.03 probability of 3x multiplier). |
| 16 | Are failure days represented? | **NO** | `ml/src/generate_full.py:31-38`: Only weekends are skipped; all simulated weekdays run identical arrival and capacity logic. |
| 17 | Are event logs generated? | **PARTIALLY (Completed citizens only)** | `ml/src/event_engine.py:90,100`: `completed_events` stores flat citizen records. Raw simulation discrete events (`ARRIVAL`, `SERVICE_END`, etc.) are discarded. |
| 18 | Are feature snapshots generated? | **NO** | Features are computed once at the citizen arrival instant and attached to the person dictionary (`ml/src/event_engine.py:68-83`). No periodic state snapshots exist. |

---

## 5. Backend Queue Audit

### Complete Lifecycle Trace:

1. **Citizen joins queue:**
   - **Route / Function:** `POST /api/v1/queues/{service_id}/join` in `backend/app/api/queues.py:70`, `join_queue()`.
   - **Token Creation:** Calls `QueueService.generate_token()` in `backend/app/services/queue_service.py:16`. Uses `with_for_update()` on the `QueueCounter` daily numbering row. Creates a new `QueueToken` with `status = TokenStatus.WAITING`.
   - **Queue Position Calculation:** Calls `QueueService.get_queue_position()` in `backend/app/services/queue_service.py:77`. Counts `QueueToken` records with matching `office_id` and `service_id` having `status == WAITING` created prior to the current token timestamp (`created_at < token.created_at`). Returns `position = count + 1`.
   - **Prediction Calculation:** Handled directly in `backend/app/api/queues.py:84-108`. Builds an ad-hoc dictionary and calls `prediction_service.predict_wait_time(pred_request)`.
   - **Prediction Storage:** Stored directly on the database row: `token.predicted_wait_minutes = int(round(raw_pred))` (`backend/app/api/queues.py:113`).
   - **WebSocket Broadcast:** Added as FastAPI background task: `ws_manager.broadcast_queue_update("JOINED", request.office_id, service_id, token.id)` (`backend/app/api/queues.py:117`).

2. **Prediction Recalculation:**
   - **CRITICAL FINDING:** **Predictions are NEVER recalculated.**
   - When subsequent tokens are called, completed, cancelled, or skipped, `broadcast_queue_update` in `backend/app/ws_manager.py:50` updates in-memory `position` and `people_ahead`, but transmits the unchanged static `t.predicted_wait_minutes` originally stored at join time.

3. **Staff Token Lifecycle Operations:**
   - **Call Next Token:** `POST /api/v1/queues/{office_id}/{service_id}/call-next` in `backend/app/api/queues.py:130`. Calls `queue_service.call_next_token()` (`backend/app/services/queue_service.py:98`), which locks the oldest WAITING token and transitions it to `TokenStatus.CALLED` (`called_at = now`). Broadcasts `"CALLED"`.
   - **Start Serving:** `POST /api/v1/tokens/{token_id}/start` in `backend/app/api/tokens.py:64`. Calls `queue_service.start_serving()` (`backend/app/services/queue_service.py:141`), transitioning `CALLED -> SERVING` (`serving_at = now`). Broadcasts `"STARTED"`.
   - **Complete Serving:** `POST /api/v1/tokens/{token_id}/complete` in `backend/app/api/tokens.py:76`. Calls `queue_service.complete_token()` (`backend/app/services/queue_service.py:151`), transitioning `SERVING -> COMPLETED` (`completed_at = now`). Broadcasts `"COMPLETED"`.
   - **Skip Token:** `POST /api/v1/tokens/{token_id}/skip` in `backend/app/api/tokens.py:88`. Calls `queue_service.skip_token()` (`backend/app/services/queue_service.py:171`), transitioning `CALLED -> SKIPPED` (`skipped_at = now`). Broadcasts `"SKIPPED"`.
   - **Cancel Token:** `POST /api/v1/tokens/{token_id}/cancel` in `backend/app/api/tokens.py:43`. Calls `queue_service.cancel_token()` (`backend/app/services/queue_service.py:161`), transitioning `WAITING -> CANCELLED` (`cancelled_at = now`). Broadcasts `"CANCELLED"`.

4. **Counter State Influence on Queue:**
   - **ZERO INFLUENCE:** In the backend, calling, starting, completing, or skipping a token does **not interact with any counter entity whatsoever**.
   - The token record (`backend/app/models/token.py:15-41`) does not have a `counter_id` foreign key. Staff employees are not bound to counters.

---

## 6. Counter Architecture Audit

### Simulator vs. Backend Comparison:

| Architectural Requirement | Backend Implementation | Simulator Implementation | Architectural Status |
|---|---|---|---|
| Real Counter entities | **NO** (Table `queue_counters` is solely a ticket sequence number generator) | **NO** (Only an integer `counters_free`) | **MISMATCH** |
| Counter Status: `ACTIVE` \| `UNAVAILABLE` | **NO** (No counter status column or table exists) | **NO** (Only scalar free integer) | **MISMATCH** |
| Assigned Service ID | **NO** (No service assignment to physical counters) | **NO** (Any counter takes any service) | **MISMATCH** |
| Service Eligibility | **NO** | **NO** | **MISMATCH** |
| `status_changed_at` timestamp | **NO** | **NO** | **MISMATCH** |
| One counter serves one service at a time | **NO** (Staff members are not linked to counters) | **NO** (All counters pull from single office FIFO) | **MISMATCH** |
| Counter linked to token being served | **NO** (`queue_tokens` lacks `counter_id`) | **NO** (Citizen only has `office_id` & `service_id`) | **MISMATCH** |

### Critical Finding on `QueueCounter`:
The model `backend/app/models/queue_counter.py:5-16` is named `QueueCounter`, but its schema is:
```python
class QueueCounter(Base):
    id = Column(Integer, primary_key=True)
    office_id = Column(Integer, ForeignKey("offices.id"))
    service_id = Column(Integer, ForeignKey("services.id"))
    date = Column(Date, default=date.today)
    current_number = Column(Integer, default=0)
```
This is **strictly an auto-incrementing ticket sequence generator** for issuing daily sequential numbers (`RJT-LL-001`, `002`), **not a desk or counter entity**. Neither the backend nor the simulator has a physical Counter model.

---

## 7. Feature Engineering Audit

### Feature Consistency Table:

| Feature Name | Expected Type | Training Source | Backend Source | Hard-Coded in Backend? | Definition Consistent? |
|---|---|---|---|---|---|
| `office_id` | `str` | `ml/src/arrival_process.py:38` (`RJT`, `AMD`, `STV`) | `backend/app/api/queues.py:92`: `office.code` | No | **Consistent** |
| `service_id` | `str` | `ml/src/arrival_process.py:39` (`LL`, `DL_RENEWAL`, `VEH_REG`) | `backend/app/api/queues.py:93`: `service.code` | No | **Consistent** |
| `hour` | `int` | `ml/src/event_engine.py:70`: `ts.hour` (simulated event timestamp) | `backend/app/api/queues.py:94`: `datetime.datetime.now().hour` | No | **Partially** (Server local time vs simulated UTC) |
| `day_of_week` | `int` | `ml/src/event_engine.py:71`: `ts.dayofweek` (0=Mon..6=Sun) | `backend/app/api/queues.py:95`: `datetime.datetime.now().weekday()` | No | **Consistent** |
| `people_ahead` | `int` | `ml/src/event_engine.py:73`: `self.queue_engine.length(office_id)` (**Office-wide FIFO!**) | `backend/app/api/queues.py:96`: `position - 1` (**Per-service queue!**) | No | **INCONSISTENT** (Training uses office-wide total; backend uses per-service count) |
| `queue_length` | `int` | `ml/src/event_engine.py:74`: Set to `people_ahead` (**Identical duplicate**) | `backend/app/api/queues.py:97`: `position` (`people_ahead + 1`) | No | **INCONSISTENT** (Training: `queue_length == people_ahead`. Backend: `queue_length == people_ahead + 1`) |
| `active_counters` | `int` | `ml/src/event_engine.py:75`: `total_counters` (static office capacity: 3, 4, 5) | `backend/app/api/queues.py:98`: `office.counters` (static database column) | Semi-static | **Consistent but flawed** (Neither reflects actual active/operational counters) |
| `arrival_rate_15m` | `int` | `ml/src/event_engine.py:48-49`: Real count of simulated arrivals in last 15 min across office | `backend/app/api/queues.py:99`: `5` (`# Mock value as actual historical rate is complex`) | **YES (`5`)** | **INCONSISTENT** |
| `counter_load` | `float` | `ml/src/event_engine.py:77`: `people_ahead / max(1, total_counters)` | `backend/app/api/queues.py:100`: `position / max(1, office.counters)` | No | **Minor mismatch** (`position` vs `people_ahead` in numerator) |
| `historical_avg_service_minutes` | `float` | `ml/src/event_engine.py:78`: Config mean service duration | `backend/app/api/queues.py:101`: `service.mean_minutes` | No | **Consistent** |
| `service_time_std` | `float` | `ml/src/event_engine.py:79`: Config std service duration | `backend/app/api/queues.py:102`: `service.std_minutes or 0.0` | No | **Consistent** |
| `no_show_rate` | `float` | `ml/src/event_engine.py:58,80`: `base_no_show * daily_var_ns * is_peak` (Stochastic value) | `backend/app/api/queues.py:103`: `0.05` (Hard-coded constant) | **YES (`0.05`)** | **INCONSISTENT** |

### Shared Feature Contract Status:
- **DOES NOT EXIST.** There is no `shared/features.py` file or shared module. Feature extraction in the backend is written manually in `backend/app/api/queues.py:91-105`.

---

## 8. ML Training & Evaluation Audit

1. **Exact Prediction Target:**
   - Target variable: `actual_wait_minutes`.
   - Computed as `(service_start_time - arrival_time).total_seconds() / 60.0`. Valid only on completed citizens (no-shows and cancellations are filtered out: `ml/feature_engineering.py:30-33`).
2. **Current Trained Model:**
   - Architecture: Scikit-learn `RandomForestRegressor` embedded in a `Pipeline` with `ColumnTransformer` (OneHotEncoder for `['office_id', 'service_id']`, remainder passthrough).
   - Tuned Hyperparameters: `n_estimators = 300`, `max_depth = None`, `min_samples_leaf = 4`, `max_features = 'sqrt'`, `random_state = 42` (`ml/models/random_forest_metadata.json:18-23`).
   - Size on disk: **723,153,418 bytes (~723 MB)** (`ml/models/random_forest_wait_time_pipeline.joblib`).
3. **Features Expected:**
   - Exactly 12 features in order: `['office_id', 'service_id', 'hour', 'day_of_week', 'people_ahead', 'queue_length', 'active_counters', 'arrival_rate_15m', 'counter_load', 'historical_avg_service_minutes', 'service_time_std', 'no_show_rate']`.
4. **Data Leakage & Collinearity:**
   - Future information leakage: Forbidden target columns (`service_start_time`, `actual_service_minutes`, etc.) were properly excluded.
   - Collinearity artifact: In the training set, `queue_length` is identical to `people_ahead` ($r = 1.0$), and `counter_load = people_ahead / active_counters`. Together, these three features account for **88.48% of total tree split importance** (`ml/models/RANDOM_FOREST_REPORT.md:234`).
   - Critical failure in validation: The robustness report (`ml/data/synthetic/validation/ROBUSTNESS_VALIDATION_REPORT.md:189`) explicitly reports: `Counter relationship (increasing counters -> decreasing pred): False`. The model failed this basic monotonicity test because counters were constant per office.
5. **Split Methodology:**
   - Chronological split: Sorted by `(date, arrival_time)` on 180,809 valid modeling rows. Train: 70% (126,566 rows), Val: 15% (27,121 rows), Test: 15% (27,122 rows) (`ml/data/synthetic/splits/BASELINE_REPORT.md:8-16`).
6. **Training vs. Backend Feature Schema Match:**
   - Key names match, but underlying distributions clash fundamentally (office-wide queue in training vs. per-service in backend; hard-coded mock values in backend).
7. **Meaningfulness of Current Metrics under PRAVAAH Redesign:**
   - **NOT MEANINGFUL.** The reported test metrics (MAE = 3.96 min, $R^2 = 0.962$) are entirely artifacts of an office-wide FIFO queue with fixed counter capacities. When per-service queues and dynamic eligible counter reassignments are introduced, this model will be completely invalid.
8. **Existing Artifacts to Preserve:**
   - Frozen dataset: `ml/data/synthetic/frozen/queue_events_full.parquet` and its checksum manifest `ml/data/synthetic/frozen/checksums.sha256`.
   - Evaluation reports in `docs/` and `ml/data/synthetic/` (record of development history).
   - The ~723 MB joblib model should eventually be retired and replaced once the new simulator generates valid per-service training data.

---

## 9. Prediction / ETA Audit

- **Point ETA:** Supported. The model returns a single float wait-time estimate, rounded to an integer in the token record (`backend/app/api/queues.py:107`).
- **Prediction Range / Uncertainty / Quantiles:** **NOT SUPPORTED.** The model does not produce prediction intervals, quantiles (e.g., P10, P90), or standard deviation estimates.
- **Residual Analysis at Runtime:** Not supported. Residuals were analyzed offline only (`ml/data/synthetic/validation/ROBUSTNESS_VALIDATION_REPORT.md`).
- **Calibration:** Not calibrated; raw Random Forest predictions are used directly.
- **Prediction Updates:** Not supported. Predictions are calculated strictly once when joining the queue. As tokens ahead are served, the remaining citizen's predicted wait time is never updated.

---

## 10. Intervention Simulation Feasibility

Can the current simulator architecture support PRAVAAH's intervention engine requirements?

| Capability | Current Simulator Support | Required Changes |
|---|---|---|
| Start from live state snapshot | **NO** (`run_day` hard-resets all state to empty) | Add state initialization API taking live queue tokens and counter busy/finish times. |
| Deep-clone state for candidate trials | **NO** (State is spread across objects with no copy/serialization) | Design a lightweight, copyable simulator state struct/dataclass. |
| Run candidate actions (e.g. reassign counter) | **NO** (No counter assignments exist) | Implement counter entity assignment to specific per-service queues. |
| Reassignment windows (15, 30, 60 mins) | **NO** (Runs whole 9-17 day only) | Implement time-bounded simulation horizon (stop after $H$ simulated minutes). |
| Configurable switching delay | **NO** | Add `switching_delay_minutes` during which counter status is transitionally unavailable. |
| Identical seeds across candidate runs | **NO** (Global numpy/random state) | Pass explicit `np.random.Generator` or seeded PRNG to each candidate run. |
| Evaluate total waiting person-minutes | **NO** (Only records individual waits for completed tokens) | Track integral of queue length $\int Q(t) dt$ or sum of all waiting person-minutes. |
| Evaluate P90 wait per service | **NO** | Collect per-service wait times within the horizon and compute 90th percentiles. |
| Constraint checks & winner logging | **NO** | Add constraint validator (e.g., no service P90 exceeds limit) and comparison log. |

---

## 11. Performance Feasibility

### Facts & Likely Bottlenecks:
1. **Pandas & Timestamp Overhead:** The current simulator creates `pd.Timestamp` and `pd.Timedelta` objects on every event (`ml/src/arrival_process.py:34`). In Python, pandas timestamp operations are 50–100x slower than raw float/integer minutes.
2. **Intervention Simulation Horizon Math:** Evaluating an intervention requires running 3 candidate policies $\times$ ~30 replications = 90 runs over a 60-minute horizon. In a 60-minute horizon across 3 services with ~50 arrivals/day/service, an office receives roughly 15–25 total arrivals. $90 \text{ runs} \times 25 \text{ events} \approx 2,250 \text{ total events}$. If implemented using pure Python integer/float minutes and `heapq`, **2,250 events will execute in well under 100 milliseconds**.
3. **Model Loading Overhead:** The Random Forest model file is **723 MB**. Loading this model takes 1.5–3 seconds and consumes ~1 GB RAM. Repeating model loading inside simulation runs would immediately freeze the process. Model inference must remain decoupled from the discrete-event intervention simulator.
4. **Database Isolation:** The intervention simulator must run purely in-memory on a snapshot. It must **not** make database queries during the 90 replication iterations.

---

## 12. Citizen Arrival / Travel Audit

- **Checked-in / Arrived State:** **NOT SUPPORTED.** The token status enum (`backend/app/models/token.py:7-14`) has only `WAITING`, `CALLED`, `SERVING`, `COMPLETED`, `CANCELLED`, `SKIPPED`. When a token is created, it enters `WAITING` immediately.
- **Citizen Travel Time:** **NOT SUPPORTED.** No citizen location, travel duration, or arrival window fields exist on `User`, `QueueToken`, or API schemas.
- **Grace Periods & Late Arrival Handling:** **NOT SUPPORTED.** When an employee calls `call-next`, status changes to `CALLED`. If the citizen is absent, the employee can immediately call `/skip`. There is no grace period timer.
- **Skipped Token Behavior:** In `backend/app/services/queue_service.py:125`, `TokenStatus.SKIPPED: []` is a **terminal state**. A skipped citizen cannot reclaim their queue spot, defer their token, or rejoin without generating a brand new token at the end of the queue.

---

## 13. Security Audit

| Finding | Severity | Description & Location |
|---|---|---|
| Insecure Fallback JWT Secret | **CRITICAL** | `backend/app/core/config.py:8`: `JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-key-for-development")`. If `.env` is absent (and `backend/.env` is currently missing), the backend runs with a publicly known secret. |
| Unauthenticated Public Prediction Endpoint | **CRITICAL** | `backend/app/api/predictions.py:7-27`: `/api/v1/predictions/wait-time` has no authentication dependency. Any unauthenticated caller can send arbitrary inputs and trigger heavy ML inference. |
| Unscoped Employee Authorization | **HIGH** | In `backend/app/api/queues.py` and `backend/app/api/tokens.py`, `require_employee` only verifies `user.role in [employee, admin]`. Employees have no office or service restrictions and can call, start, complete, or skip tokens for **any office and service**. |
| Missing CORS Middleware | **HIGH** | `backend/app/main.py`: Zero CORS configuration. Any external web frontend (React on `localhost:3000` or `localhost:5173`) will be blocked by browser CORS policy. |
| JWT Token in WebSocket URL Query Parameter | **HIGH** | `backend/app/api/deps.py:56`: `token = websocket.query_params.get("token")`. Passing bearer tokens in query strings leaks JWTs into browser history, server logs, and proxies. |
| DB Exception Leakage | **MEDIUM** | `backend/app/main.py:42`: Endpoint `/health/db` catches `Exception as e` and returns `raise HTTPException(status_code=500, detail=str(e))`, leaking database connection strings and internal error details. |
| No Rate Limiting | **MEDIUM** | No rate limiting middleware on `/auth/login`, `/auth/register`, `/queues/{id}/join`, or `/predictions/wait-time`. Vulnerable to brute-force and resource exhaustion against the 723MB model. |
| Local vs. UTC Timezone Mismatch | **MEDIUM** | `backend/app/services/queue_service.py:14` uses UTC for DB timestamps, but `backend/app/api/queues.py:94` passes `datetime.datetime.now().hour` (system local time) into the ML model. |
| Hard-Coded Office/Service Enums in Validation | **LOW** | `backend/app/api/predictions.py:9-15`: Valid offices (`RJT`, `AMD`, `STV`) and services are hardcoded in static sets rather than checked against the database. |
| Secure Password Hashing & ORM Parametrization | **INFO** | Password hashing securely uses bcrypt with salt (`backend/app/core/security.py:6-15`). SQLAlchemy ORM parametrizes queries safely. |

---

## 14. Component Classification: KEEP / MODIFY / REWRITE / REMOVE / UNKNOWN

| Component | Path / Area | Classification | Detailed Rationale |
|---|---|---|---|
| **`QueueEngine`** | `ml/src/queue_engine.py` | **REWRITE** | Currently merges all services into one office-wide FIFO queue. Must be completely rewritten to maintain independent per-service queues. |
| **`CounterManager`** | `ml/src/counter_manager.py` | **REWRITE** | Currently a scalar integer allocator. Must be rewritten into an entity manager tracking individual Counter instances with eligibility sets, assigned service, and status (`ACTIVE`/`UNAVAILABLE`). |
| **`EventEngine`** | `ml/src/event_engine.py` | **REWRITE** | Lacks state snapshot initialization, time horizons, counter assignments, and independent seeded RNGs. Must be rewritten as a clean discrete-event engine suitable for both training generation and intervention rollouts. |
| **`ArrivalProcess`** | `ml/src/arrival_process.py` | **MODIFY** | The core hourly weighting approach is good, but it must generate per-service arrival streams and support time-varying burst events. |
| **`ServiceProcess`** | `ml/src/service_process.py` | **KEEP / MODIFY** | The lognormal duration logic is statistically sound; modify to accept counter efficiency/speed variations if needed. |
| **Full Dataset Generator** | `ml/src/generate_full.py` | **REWRITE** | Must be rewritten to drive the rewritten per-service simulator once ready. |
| **Feature Engineering Script** | `ml/feature_engineering.py` | **REWRITE** | Offline script with duplicate features (`queue_length` = `people_ahead`). Must be replaced by a pure shared feature extractor. |
| **Trained Random Forest Pipeline** | `ml/models/random_forest_wait_time_pipeline.joblib` | **REMOVE / RETRAIN** | Trained on invalid office-wide queue simulation; unwieldy 723 MB size. Must be retrained on valid PRAVAAH synthetic data. |
| **`PredictionService`** | `backend/app/services/prediction_service.py` | **MODIFY** | Clean singleton wrapper; needs modification to call the shared feature contract and output ETA ranges/uncertainty. |
| **`QueueService`** | `backend/app/services/queue_service.py` | **MODIFY** | Token generation with row locking and state transitions are solid; modify to associate `call_next_token` with an explicit counter. |
| **`QueueCounter` Model** | `backend/app/models/queue_counter.py` | **KEEP (Clarify)** | Keep for daily token number generation (`RJT-LL-001`), but do not treat as a desk counter entity. |
| **Physical Counter Model** | `backend/app/models/counter.py` | **CREATE (NEW)** | Real `Counter` entity (`id`, `office_id`, `name`, `status`, `assigned_service_id`, `status_changed_at`) and `counter_services` eligibility table. |
| **Ad-hoc Feature Assembly** | `backend/app/api/queues.py:91-105` | **REWRITE** | Replace hard-coded values (`5`, `0.05`) with a single pure shared feature function. |
| **Queue Broadcasts** | `backend/app/ws_manager.py` | **MODIFY** | Connection pool is solid; modify to update remaining citizens with recalculated ETAs when tokens advance. |
| **Authentication & Security** | `backend/app/core/security.py`, `backend/app/api/auth.py` | **KEEP / MODIFY** | Bcrypt and JWT logic are production-quality; modify to scope employees to offices and add CORS. |
| **Admin Analytics API** | `backend/app/api/analytics.py` | **MODIFY** | Add queue stability metrics ($\lambda, \mu$, STABLE/WATCH/UNSTABLE) and counter utilization analytics. |

---

## 15. Architecture Mismatches

The discrepancies between the current repository and the PRAVAAH target design are prioritized by severity:

### Priority 1: Queue & Counter Domain Model Mismatch (Critical Blocker)
- **Proposed PRAVAAH Design:** Individual counter entities (`id`, `office_id`, `status: ACTIVE | UNAVAILABLE`, `assigned_service_id`, `status_changed_at`), service eligibility sets, and one independent queue per service per office.
- **Current Codebase:** No counter entity in backend or simulator. Simulator runs a single office-wide FIFO queue for all services.
- **Impact:** Invalidates the core ML training dataset and makes intervention comparison impossible.

### Priority 2: Absence of Shared Feature Contract (High)
- **Proposed PRAVAAH Design:** A pure, deterministic shared feature function (`shared/features.py`) used by both simulator and live backend with zero hard-coded values.
- **Current Codebase:** Simulator computes features in `ml/src/event_engine.py:68-83`; backend assembles an ad-hoc dictionary in `backend/app/api/queues.py:91-105` with hard-coded values (`arrival_rate_15m = 5`, `no_show_rate = 0.05`).

### Priority 3: Simulator Incapable of Intervention Simulation (High)
- **Proposed PRAVAAH Design:** Simulator accepts live state snapshots, deep-clones state, applies counter reassignment policies across 15/30/60 min horizons, and runs identical seeds to compare:
  1. No action
  2. Naive longest-queue policy
  3. PRAVAAH optimizer.
- **Current Codebase:** Simulator only runs whole-day simulations from empty queues; cannot clone state, reassign counters, or run horizon-bounded replications.

### Priority 4: Stale, Static Point ETA Predictions (Medium)
- **Proposed PRAVAAH Design:** Dynamic per-citizen ETA intervals/uncertainty, updated as queue conditions evolve.
- **Current Codebase:** Single point prediction generated once at queue join; never recalculated as the queue advances.

### Priority 5: Missing Security & Operational Scoping (Medium)
- **Current Codebase:** Missing CORS middleware; unauthenticated `/predictions/wait-time`; unscoped employee permissions; hard-coded default JWT secret.

---

## 16. Implementation Dependencies

A strict dependency graph showing the order of operations for implementing PRAVAAH:

```
[1. Counter & Queue Domain Models]
     (Counters as entities, eligibility, per-service queues in Backend DB & Simulator)
                           │
                           ▼
[2. Shared Pure Feature Contract]
     (shared/features.py: pure, deterministic, state-derived, zero hard-coding)
                           │
                           ▼
[3. Redesigned Simulator Core]
     (Per-service queues, entity counters, state snapshotting, horizon limits, seeded PRNG)
                           │
                           ▼
[4. Correct Synthetic Dataset Generation & Freezing]
     (Generate valid multi-service queue events from new simulator)
                           │
                           ▼
[5. Retrain ML Model & Uncertainty Calibration]
     (Fast per-citizen point ETA + prediction uncertainty/intervals)
                           │
                           ▼
[6. Backend Integration & ETA Recalculation]
     (Hook shared feature contract into join/websocket flows, update predictions dynamically)
                           │
                           ▼
[7. Intervention Simulator & Policy Comparison Engine]
     (Snapshot state -> Run 3 candidates x 30 reps with same seeds -> Minimize Person-Minutes)
                           │
                           ▼
[8. Stability Metrics & Admin Analytics API]
     (Expose λ, μ, STABLE/WATCH/UNSTABLE, time-to-congestion, and intervention recommendations)
                           │
                           ▼
[9. Frontend Integration & Security Hardening]
     (React UI, CORS middleware, scoped employee desks, arrival window displays)
```

---

## 17. Risks / Open Questions

1. **Database Counter Schema Migration:** How should physical counters be seeded for existing offices (`RJT`, `AMD`, `STV`) without breaking existing tests that expect `office.counters` as a plain integer?
2. **Employee-to-Counter Mapping:** When an employee logs in, do they explicitly claim a physical counter desk (e.g. Counter 1), or does the system assign the next available active counter?
3. **Arrival Rate $\lambda$ Windowing in Live Backend:** What rolling window (e.g., 15 minutes, 30 minutes, or 60 minutes) should the live backend use to calculate current arrival rate $\lambda$ per service from PostgreSQL `queue_tokens`?
4. **Intervention Switching Delay Value:** What switching delay penalty (e.g., 2 minutes, 5 minutes) should apply when a counter transitions between different services?
5. **Model Artifact Size Constraint:** The current 723 MB Random Forest model is unacceptably bloated for hackathon deployment. Can we restrict tree depth (`max_depth=12`, `n_estimators=100`) or use LightGBM/HistGradientBoosting to reduce artifact size to < 20 MB?

---

## 18. Recommended Next Step

### **RECOMMENDED NEXT STEP:**
**Step 2: Formalize the Counter & Queue Domain Model Specification (Data Models + Shared Contract Interface).**

Specifically, create the architectural contract and SQLAlchemy/dataclass schemas defining:
1. `Counter` entity (`id`, `office_id`, `name`, `status`, `assigned_service_id`, `status_changed_at`),
2. `CounterService` eligibility mapping,
3. The exact signature and input state dataclass for `shared/features.py`.

*(Do not write implementation code or execute migrations until this specification is finalized).*
