# PRAVAAH — Backend API Contract & Integration Guide

**Version:** 1.0 (Step 2E.5A Backend Completion)  
**Base URL:** `http://localhost:8000` (or configured deployment host)  
**WebSocket URL:** `ws://localhost:8000/ws/queues/{office_id}/{service_id}`  
**Architecture Contract:** Invalidation-only real-time hints; REST snapshots remain the single source of truth.

---

## 1. Overview & Operational Loop

PRAVAAH backend provides a complete operational decision and control loop:

```
Observe (Snapshot/Features) 
   → Predict (Hybrid ML + Empirical P10–P90) 
   → Detect (Queue Pressure / Stability Bands) 
   → Simulate (Discrete-Event Simulation Look-Ahead) 
   → Recommend (Baseline vs. Naive vs. Optimized) 
   → Apply (Live Atomic Reassignment) 
   → Verify (Immediate State & Invalidation Events) 
   → Measure (Realized Waiting Times & Fleet Throughput)
```

### Key Domain Distinctions & Step 2E.5B Guarantees

1. **Switching Delay Distinction:**
   - In discrete-event simulation look-ahead (`POST /api/v1/interventions/evaluate`), a **3.0-minute counter switching delay** is explicitly modeled to assess transition costs before recommending a candidate.
   - In live database execution (`POST /api/v1/interventions/apply`), the counter reassignment is **immediate** (`assigned_service_id` and `status_changed_at` are persisted immediately in an atomic transaction). The 3-minute delay is purely an evaluation look-ahead penalty.
2. **Busy Counter Reassignment & Unavailability Safety:**
   - Applying a valid reassignment or marking a counter `UNAVAILABLE` succeeds even when the counter is currently serving an active token.
   - The active token remains associated with its original service and counter, and can be completed normally through `/api/v1/queues/{token_id}/complete`.
   - The counter will not accept any new tokens while the current token is in progress.
   - Once the active token completes, subsequent calls strictly respect the new state: serving the new assigned service (or blocked if `UNAVAILABLE`).
3. **Fluid Time-to-Congestion Metric:**
   - Computed transparently in the admin summary service using the fluid backlog approximation:
     $$\text{TTC}_{\text{minutes}} = \frac{(Q_{\text{threshold}} - Q) \times 60.0}{\lambda - \mu}$$
   - Method label: `time_to_congestion_method: "fluid_approximation"`.
   - Arrival rate $\lambda$ (`arrival_rate_60m`) and service capacity $\mu$ (`service_capacity_mu`) are both in **citizens/hour**.
   - Configurable backlog threshold: default `15` tokens (configurable via `CONGESTION_QUEUE_THRESHOLD` or `?congestion_threshold=` query param).
   - Clean boundary handling: returns `0.0` if already congested ($Q \ge Q_{\text{threshold}}$), returns `null` if stable or draining ($\lambda \le \mu$). Never returns negative minutes, NaN, or infinity.
4. **WebSocket Invalidation Hints:**
   - WebSockets do **not** transmit full token lists or citizen PII. They broadcast compact, post-commit invalidation hints (`queue.updated` and `counter.status_changed`).
   - Clients treat these hints as triggers to refetch REST snapshots.
5. **CORS Development Defaults:**
   - Out-of-the-box support for `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:8080`, `http://localhost:4173`.
   - Override or extend via `CORS_ORIGINS` environment variable (comma-separated). Credentials (`allow_credentials=True`) are enabled. Wildcards and `Origin: null` are rejected.

---

## 2. Authentication & Authorization

All endpoints except public registration/login and citizen status require a Bearer JWT in the `Authorization` header:

```http
Authorization: Bearer <access_token>
```

### Endpoints

#### POST `/api/v1/auth/register`
- **Purpose:** Citizen registration
- **Role:** Public
- **Request Body:**
  ```json
  {
    "email": "citizen@example.com",
    "password": "secretpassword",
    "role": "citizen"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "id": 1,
    "email": "citizen@example.com",
    "role": "citizen",
    "is_active": true
  }
  ```

#### POST `/api/v1/auth/login`
- **Purpose:** User authentication (OAuth2 Password flow)
- **Role:** Public
- **Request Body:** Form data (`username=<email>&password=<password>`) or JSON
- **Response (200 OK):**
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "token_type": "bearer"
  }
  ```

#### GET `/api/v1/auth/me`
- **Purpose:** Retrieve current authenticated profile
- **Role:** Any authenticated user
- **Response (200 OK):**
  ```json
  {
    "id": 1,
    "email": "admin@gov.in",
    "role": "admin",
    "is_active": true
  }
  ```

---

## 3. Citizen Screen APIs

### 3.1. Public Queue Status
- **Method & Path:** `GET /api/v1/queues/{office_id}/{service_id}/status`
- **Purpose:** Fetch waiting queue length and active desk count prior to joining
- **Role:** Citizen / Public
- **Mutating:** No (Read-only)
- **Invalidation Emitted:** No
- **Response (200 OK):**
  ```json
  {
    "waiting_count": 8,
    "active_counters": 2
  }
  ```
- **Error Codes:** `404 Not Found` (invalid office or service ID).

### 3.2. Join Queue
- **Method & Path:** `POST /api/v1/queues/{service_id}/join`
- **Purpose:** Issue a new sequential queue token for the authenticated citizen
- **Role:** Citizen
- **Mutating:** Yes
- **Invalidation Emitted:** Yes (`type: "queue.updated"`)
- **Request Body:**
  ```json
  {
    "office_id": 1
  }
  ```
- **Response (201 Created):**
  ```json
  {
    "id": 1042,
    "token_number": "MUNI-PT-042",
    "office_id": 1,
    "service_id": 1,
    "counter_id": null,
    "status": "WAITING",
    "queue_position": 9,
    "predicted_wait_minutes": 24.5,
    "created_at": "2026-10-09T03:45:00Z",
    "called_at": null,
    "serving_at": null,
    "completed_at": null,
    "cancelled_at": null,
    "skipped_at": null
  }
  ```
- **Error Codes:** `400 Bad Request` (invalid office/service), `401 Unauthorized`.

### 3.3. Live Token ETA (Step 2E.4 Pipeline)
- **Method & Path:** `GET /api/v1/queues/{token_id}/eta`
- **Purpose:** Retrieve ML wait-time prediction and empirical P10–P90 intervals derived from live DB snapshot
- **Role:** Citizen (object-level scoped to own token) or Staff/Admin
- **Query Parameters:** `office_id` (optional, validated against token's authoritative office)
- **Mutating:** No (Read-only)
- **Invalidation Emitted:** No
- **Response (200 OK):**
  ```json
  {
    "token_id": 1042,
    "token_number": "MUNI-PT-042",
    "office_id": 1,
    "office_code": "MUNI",
    "service_id": 1,
    "service_code": "PT",
    "predicted_wait_minutes": 24.5,
    "p10_minutes": 18.0,
    "p90_minutes": 32.5,
    "prediction_source": "ML_MODEL",
    "fallback_reason": null,
    "people_ahead": 8,
    "queue_length": 8,
    "is_cold_start": false,
    "diagnostics": [],
    "snapshot_time_utc": "2026-10-09T03:46:12Z",
    "provenance": "Empirical P10–P90 prediction intervals from quantile models, not statistical confidence intervals."
  }
  ```
- **Error Codes:** `404 Not Found` (token not found), `403 Forbidden` (accessing another citizen's token), `400 Bad Request` (token not in WAITING status).

### 3.4. Token Details & Position
- **Method & Path:** `GET /api/v1/tokens/{token_id}`
- **Purpose:** Retrieve current token status
- **Role:** Citizen (token owner) / Staff / Admin
- **Response (200 OK):** `TokenResponse`

---

## 4. Employee Screen APIs

### 4.1. Queue Snapshot for Service
- **Method & Path:** `GET /api/v1/queues/{office_id}/{service_id}/tokens`
- **Purpose:** View active tokens (`WAITING`, `CALLED`, `SERVING`) for service desk dispatch
- **Role:** Employee / Admin
- **Mutating:** No
- **Response (200 OK):** Array of `TokenResponse` objects.

### 4.2. Call Next Token (Counter-Specific)
- **Method & Path:** `POST /api/v1/queues/{office_id}/{service_id}/call-next`
- **Purpose:** Call the oldest waiting token in strict FIFO order (`created_at ASC, id ASC`) and assign to physical counter
- **Role:** Employee / Admin
- **Mutating:** Yes
- **Invalidation Emitted:** Yes (`type: "queue.updated"`)
- **Request Body:**
  ```json
  {
    "counter_id": 1
  }
  ```
  *(Note: `counter_id` is optional for backward compatibility, but recommended for physical desk integrity).*
- **Server-Side Validation Invariants:**
  1. Counter exists and belongs to `office_id`.
  2. Counter status is `ACTIVE` (not `UNAVAILABLE`).
  3. Counter is idle (no active token in `CALLED` or `SERVING`).
  4. Counter is assigned to `service_id`.
  5. Counter is eligible to serve `service_id`.
  6. At least one token is in `WAITING` status.
- **Response (200 OK):**
  ```json
  {
    "id": 1001,
    "token_number": "MUNI-PT-001",
    "office_id": 1,
    "service_id": 1,
    "counter_id": 1,
    "status": "CALLED",
    "queue_position": 0,
    "called_at": "2026-10-09T03:50:00Z"
  }
  ```
- **Error Codes:**
  - `404 Not Found`: No waiting tokens in queue, or counter does not exist.
  - `409 Conflict`: Counter is already busy with an active token.
  - `400 Bad Request`: Counter is UNAVAILABLE, not assigned to service, or ineligible.

### 4.3. Token Lifecycle Mutations
All endpoints require `Employee` or `Admin` role, mutate token state, and emit `queue.updated` invalidation hints.

- **Start Serving:** `POST /api/v1/tokens/{token_id}/start`
  - Moves token from `CALLED` → `SERVING`.
- **Complete Token:** `POST /api/v1/tokens/{token_id}/complete`
  - Moves token from `SERVING` → `COMPLETED`. Releases counter to idle state.
- **Skip Token:** `POST /api/v1/tokens/{token_id}/skip`
  - Moves token from `CALLED` → `SKIPPED`. Releases counter.
- **Cancel Token:** `POST /api/v1/tokens/{token_id}/cancel`
  - Moves token from `WAITING` → `CANCELLED`.

---

## 5. Admin Command-Center APIs

### 5.1. Admin Operations Summary
- **Method & Path:** `GET /api/v1/admin/offices/{office_id}/summary`
- **Purpose:** Complete real-time operational overview of all queues and desks in an office
- **Role:** Admin
- **Mutating:** No (Read-only)
- **Invalidation Emitted:** No
- **Response (200 OK):**
  ```json
  {
    "office_id": 1,
    "office_code": "MUNI",
    "office_name": "Municipality Office",
    "snapshot_time_utc": "2026-10-09T04:00:00Z",
    "total_waiting_tokens": 10,
    "total_active_counters": 6,
    "total_busy_counters": 5,
    "total_unavailable_counters": 1,
    "services": [
      {
        "service_id": 1,
        "service_code": "PT",
        "service_name": "Property Tax",
        "queue_length": 8,
        "assigned_active_counters": 2,
        "busy_counters": 2,
        "unavailable_counters": 0,
        "arrival_rate_30m": 12.0,
        "arrival_rate_60m": 12.0,
        "service_mean_minutes": 10.0,
        "service_std_minutes": 4.0,
        "service_capacity_mu": 12.0,
        "traffic_intensity": 1.0,
        "queue_health": "UNSTABLE",
        "time_to_congestion_minutes": 70.0,
        "time_to_congestion_method": "fluid_approximation",
        "congestion_threshold": 15
      },
      {
        "service_id": 2,
        "service_code": "WC",
        "service_name": "Water Service",
        "queue_length": 1,
        "assigned_active_counters": 2,
        "busy_counters": 2,
        "unavailable_counters": 0,
        "arrival_rate_30m": 4.0,
        "arrival_rate_60m": 4.0,
        "service_mean_minutes": 15.0,
        "service_std_minutes": 6.0,
        "service_capacity_mu": 8.0,
        "traffic_intensity": 0.5,
        "queue_health": "STABLE",
        "time_to_congestion_minutes": null,
        "time_to_congestion_method": "fluid_approximation",
        "congestion_threshold": 15
      }
    ],
    "counters": [
      {
        "counter_id": 1,
        "name": "C1",
        "assigned_service_id": 1,
        "assigned_service_code": "PT",
        "status": "ACTIVE",
        "is_busy": true,
        "current_token_id": 1001,
        "current_token_number": "MUNI-PT-001",
        "eligible_service_ids": [1, 3]
      }
    ]
  }
  ```
- **Error Codes:** `404 Not Found` (office not found), `403 Forbidden` (non-admin).

### 5.2. Scenario Loader & Reset
- **Method & Path:** `POST /api/v1/admin/offices/{office_id}/scenarios/load`
- **Purpose:** Atomically reset and initialize office demo database with a frozen simulation scenario
- **Role:** Admin
- **Mutating:** Yes (Atomic transaction)
- **Invalidation Emitted:** Yes (`type: "queue.updated"` for all affected services)
- **Request Body:**
  ```json
  {
    "scenario_id": "scenario_b_pt_pressure",
    "now": null
  }
  ```
  *(Short aliases also supported: `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"rto_healthy"`).*
- **Response (200 OK):**
  ```json
  {
    "scenario_id": "scenario_b_pt_pressure",
    "scenario_name": "Scenario B: Property Tax Queue Pressure",
    "office_id": 1,
    "office_code": "MUNI",
    "affected_service_ids": [1, 2, 3],
    "counters_loaded": 6,
    "waiting_tokens_loaded": 10,
    "busy_counters_loaded": 6,
    "loaded_at_utc": "2026-10-09T04:05:00Z"
  }
  ```
- **Error Codes:** `400 Bad Request` (invalid scenario ID or office mismatch), `404 Not Found`.

#### GET `/api/v1/admin/scenarios`
- **Purpose:** List all available scenarios and names for dropdown selection
- **Role:** Admin
- **Response (200 OK):**
  ```json
  {
    "scenarios": [
      { "scenario_id": "scenario_a_healthy", "scenario_name": "Scenario A: Healthy Operation (Baseline)" },
      { "scenario_id": "scenario_b_pt_pressure", "scenario_name": "Scenario B: Property Tax Queue Pressure" },
      { "scenario_id": "scenario_c_donor_constraint", "scenario_name": "Scenario C: Donor Service Constraints" },
      { "scenario_id": "scenario_e_counter_failure", "scenario_name": "Scenario E: Counter Failure / Capacity Loss" },
      { "scenario_id": "E_known", "scenario_name": "Scenario E (Known): Known Counter Failure at t=0" },
      { "scenario_id": "scenario_rto_healthy", "scenario_name": "Scenario RTO: Healthy Operation" }
    ]
  }
  ```

### 5.3. Intervention Evaluation (Simulation Look-Ahead)
- **Method & Path:** `POST /api/v1/interventions/evaluate`
- **Purpose:** Run pure discrete-event simulation comparing Baseline vs. Naive vs. Optimized policy without modifying DB state
- **Role:** Admin
- **Mutating:** No (Pure simulation preview)
- **Request Body:**
  ```json
  {
    "office_id": 1,
    "horizon_minutes": 60.0,
    "replications": 5,
    "base_seed": 42
  }
  ```
- **Response (200 OK):** `InterventionEvaluationResponse` with `applied: false`, `is_simulation_only: true`, and `selected_candidate`.

### 5.4. Real Intervention Apply
- **Method & Path:** `POST /api/v1/interventions/apply`
- **Purpose:** Apply real counter reassignment in the operational database
- **Role:** Admin
- **Mutating:** Yes
- **Invalidation Emitted:** Yes (`counter.status_changed` and `queue.updated` for donor and target services)
- **Request Body:**
  ```json
  {
    "office_id": 1,
    "counter_id": 5,
    "target_service_id": 1
  }
  ```
- **Validation Rules & Active-Token Guarantees:**
  - Counter must be `ACTIVE`.
  - Reassignment succeeds even if the donor counter is currently serving an active token!
  - The in-progress token remains intact under its original service and counter, and completes normally.
  - Counter must be eligible for `target_service_id`.
  - Donor service must retain at least 1 active counter.
- **Response (200 OK):**
  ```json
  {
    "applied": true,
    "office_id": 1,
    "counter_id": 5,
    "previous_service_id": 3,
    "target_service_id": 1,
    "reassigned_at_utc": "2026-10-09T04:10:00Z",
    "switching_delay_note": "Operational database reassignment is immediate. The 3.0-minute switching delay is modeled strictly in discrete-event simulation look-ahead."
  }
  ```
- **Error Codes:**
  - `400 Bad Request`: Counter is ineligible, already assigned to target service, or donor service would be left with 0 active desks.
  - `404 Not Found`: Counter or office not found.

### 5.5. Real Intervention Restore (Revert to Home Assignment)
- **Method & Path:** `POST /api/v1/interventions/restore`
- **Purpose:** Restore physical counter to its authoritative home service assignment
- **Role:** Admin
- **Mutating:** Yes
- **Invalidation Emitted:** Yes (`counter.status_changed` for both services)
- **Request Body:**
  ```json
  {
    "office_id": 1,
    "counter_id": 5
  }
  ```
- **Authoritative Home Assignments:**
  - Municipality (Office 1): C1 → PT (1), C2 → PT (1), C3 → WC (2), C4 → WC (2), C5 → BC (3), C6 → BC (3)
  - RTO (Office 2): R1 → OT (4), R2 → OT (4), R3 → DR (5)
- **Active-Token Guarantees:**
  - Restoration succeeds without corrupting in-progress or completed tokens.
- **Response (200 OK):**
  ```json
  {
    "restored": true,
    "office_id": 1,
    "counter_id": 5,
    "previous_service_id": 1,
    "restored_service_id": 3,
    "restored_at_utc": "2026-10-09T04:15:00Z",
    "message": "Counter 'C5' restored to home service 3 successfully."
  }
  ```
- **Error Codes:**
  - `400 Bad Request`: Counter is already at home assignment or UNAVAILABLE.

### 5.6. Physical Counter Availability
- **Mark Unavailable:** `POST /api/v1/counters/{counter_id}/unavailable`
  - Succeeds even if counter is currently serving an active token!
  - In-progress token remains completable through `/api/v1/queues/{token_id}/complete`.
  - Sets `status = UNAVAILABLE`. Excludes desk from `call-next`.
  - Emits single `counter.status_changed` invalidation.
- **Recover Counter:** `POST /api/v1/counters/{counter_id}/recover`
  - Sets `status = ACTIVE` without modifying service assignment.
  - Emits single `counter.status_changed` invalidation.

---

## 6. WebSocket Real-Time Invalidation Contract

### Connection Details
- **Endpoint:** `ws://localhost:8000/ws/queues/{office_id}/{service_id}`
- **Authentication:** Query parameter `?token=<access_token>` or `Authorization: Bearer <token>` handshake header.

### Invalidation Event Payload Contract
All WebSocket messages are **compact invalidation hints** without token lists or citizen PII:

```json
{
  "type": "queue.updated",
  "office_id": 1,
  "service_id": 1,
  "occurred_at": "2026-10-09T04:12:35.123456+00:00"
}
```

Or for desk availability changes:
```json
{
  "type": "counter.status_changed",
  "office_id": 1,
  "service_id": 1,
  "occurred_at": "2026-10-09T04:12:35.123456+00:00"
}
```

### Client Frontend Handling Strategy
1. On receiving any invalidation event matching the user's active screen:
   - Citizen screen: Refetch `/api/v1/queues/{office_id}/{service_id}/status` and `/api/v1/queues/{token_id}/eta`.
   - Employee screen: Refetch `/api/v1/queues/{office_id}/{service_id}/tokens`.
   - Admin command center: Refetch `/api/v1/admin/offices/{office_id}/summary`.
2. Fallback polling:
   - Poll REST endpoints every 10–15 seconds to ensure eventual consistency if WebSocket disconnects.
