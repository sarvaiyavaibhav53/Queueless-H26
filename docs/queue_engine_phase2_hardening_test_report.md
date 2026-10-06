# Queue Engine Phase 2 - Hardening & Concurrency Test Report

## 1. Environment
- **Date:** 2026-10-04
- **OS:** Windows
- **Python Version:** 3.14.7
- **FastAPI Version:** 0.142.2
- **SQLAlchemy Version:** 2.1.3
- **PostgreSQL Version:** PostgreSQL 18.6 on x86_64-windows

## 2. Phase 2 API
- **Implemented Endpoints:**
  - `POST /api/v1/queues/{service_id}/join`
  - `GET /api/v1/tokens/{token_id}`
  - `GET /api/v1/tokens/{token_id}/position`
  - `POST /api/v1/tokens/{token_id}/cancel`
  - `POST /api/v1/queues/{office_id}/{service_id}/call-next`
  - `POST /api/v1/tokens/{token_id}/start`
  - `POST /api/v1/tokens/{token_id}/complete`
  - `POST /api/v1/tokens/{token_id}/skip`
- **Request/Response Schemas:** Validated `JoinQueueRequest`, `TokenResponse`, and `PositionResponse`. No sensitive database IDs or internals are leaked to clients inappropriately. 
- **Current Authorization Status:** Staff routes are intentionally stubbed pending the core authentication phase implementation.

## 3. Transaction & Locking
- **QueueCounter Row Locking:** Utilizes PostgreSQL `FOR UPDATE` (`with_for_update()` in SQLAlchemy) to lock the sequence row during increment operations. 
- **QueueToken Locking:** Tokens are locked during state transitions (e.g., `start_serving`, `complete_token`, `call_next_token`) to prevent simultaneous modifications.
- **Race Condition Prevention:** Implemented a savepoint/nested transaction (`db.begin_nested()`) during the initial creation of a `QueueCounter` record. This guarantees that if two threads attempt to concurrently create the counter for a new day/service, one gracefully catches the `IntegrityError` and proceeds with the existing lock, preventing crashes.
- **Transaction Boundaries:** FastAPI dependencies yield a fresh `SessionLocal` per request, strictly bound to the HTTP lifecycle, ensuring rollback on any unhandled exception.

## 4. Concurrency Results
- **Concurrent queue joins:**
  - Requests: 20
  - Successful: 20
  - Duplicate tokens: 0
  - Failed: 0
  - Result: PASS
- **Concurrent call-next:**
  - Requests: 10
  - Successful calls: 10
  - Duplicate token assignments: 0
  - Result: PASS
- **Mixed Concurrency (Join + Call-Next):**
  - Requests: 20 mixed requests
  - Successful: 20 (including appropriate 404s when queues were empty)
  - Duplicate token assignments: 0
  - Result: PASS

## 5. Queue State Tests
- **Valid Transitions:** 
  - `WAITING -> CALLED -> SERVING -> COMPLETED` (PASS)
  - `WAITING -> CANCELLED` (PASS)
  - `CALLED -> SKIPPED` (PASS)
- **Invalid Transitions Rejected:**
  - `WAITING -> COMPLETED` (PASS - Rejected)
  - `WAITING -> SERVING` (PASS - Rejected)
  - `CALLED -> CANCELLED` (PASS - Rejected)
  - `SERVING -> SKIPPED` (PASS - Rejected)
  - `COMPLETED -> CANCELLED` (PASS - Rejected)

## 6. Transaction Rollback
- **Failed Join Request Rollback:** Tested by joining an invalid service ID. The `QueueCounter` successfully verified to remain un-incremented, and the session continued processing valid requests without corruption. (PASS)
- **Invalid State Transition Rollback:** Attempting invalid transitions throws a 400 without persisting any partial updates. (PASS)

## 7. Regression Tests
- Phase 1 Queue Tests: 5 passed
- Database Tests: 3 passed
- Phase 2 API Tests: 8 passed
- Hardening Tests: 5 passed (Includes Real Concurrency)
- ML Tests: 15 blocked / 0 passed / 0 failed (Blocked due to Windows Application Control Policy affecting SciPy DLLs)

## 8. Final Summary
- **Total passed:** 21
- **Total failed:** 0
- **Total blocked:** 15
- **Total skipped:** 0
- **Warnings:** Minor Pydantic v2 deprecation warnings in the test runner.
- **Known limitations:** Real authorization is stubbed; ML integration degrades gracefully to `null` due to localized DLL blockage.
- **Remaining risks:** None in the queue state or integrity domain.

## Final Verdict
**PASS WITH WARNINGS**
