# QueueLess Backend Manual/API Test Report

## 1. Test Information
- **Project Name:** QueueLess Government Office
- **Test Date:** 2026-10-04
- **Operating System:** Windows
- **Python Version:** 3.14.7
- **FastAPI Version:** 0.142.2
- **SQLAlchemy Version:** 2.1.3
- **PostgreSQL Version:** PostgreSQL 18.6
- **Testing Method:** Live HTTP/REST API validation using Python URLLib against running Uvicorn server
- **API/Swagger URL:** `http://127.0.0.1:8000/docs` (`http://127.0.0.1:8000/openapi.json`)

## 2. Scope
This manual test covers:
- FastAPI endpoints execution
- PostgreSQL integration and data persistence
- RTO queue lifecycle (creation, tracking, cancellation)
- Citizen operations (Joining queue, checking position)
- Staff operations (Calling next token, starting service, completing, skipping)
- ML prediction integration behavior under load
- Token position recalculation dynamic accuracy
- State machine transition validation
- Pydantic/FastAPI error handling and validation
- Persistence consistency
- API response accuracy

## 3. Endpoint Inventory

| Method | Endpoint | Purpose | Tested | Result |
| ------ | -------- | ------- | ------ | ------ |
| GET | `/api/v1/predictions/wait-time` | (Not publicly tested via queue test) | No | N/A |
| POST | `/api/v1/queues/{service_id}/join` | Citizen joins a specific queue | Yes | PASS |
| POST | `/api/v1/queues/{office_id}/{service_id}/call-next` | Staff assigns next token | Yes | PASS |
| GET | `/api/v1/tokens/{token_id}` | Fetch full token details | Yes | PASS |
| GET | `/api/v1/tokens/{token_id}/position` | Fetch current wait position | Yes | PASS |
| POST | `/api/v1/tokens/{token_id}/cancel` | Citizen cancels own token | Yes | PASS |
| POST | `/api/v1/tokens/{token_id}/start` | Staff starts serving | Yes | PASS |
| POST | `/api/v1/tokens/{token_id}/complete` | Staff finishes serving | Yes | PASS |
| POST | `/api/v1/tokens/{token_id}/skip` | Staff skips token | Yes | PASS |
| GET | `/health/db` | Database healthcheck | No | N/A |

## 4. Test Case Results

| ID | Test | Expected | Actual | Status |
| -- | ---- | -------- | ------ | ------ |
| TC01 | Citizen 1 Joins | 201 Created, predicted_wait_minutes populated | 201 Created, `predicted_wait_minutes=5.0`, `queue_position=1` | PASS |
| TC02 | Citizen 2 Joins | 201 Created, pos=2, predicted wait > prev | 201 Created, `predicted_wait_minutes=9.0`, `queue_position=2` | PASS |
| TC03 | Citizen 3 Joins | 201 Created, pos=3, predicted wait > prev | 201 Created, `predicted_wait_minutes=12.0`, `queue_position=3` | PASS |
| TC04 | Citizen token lookup | 200 OK, returns correct token data | 200 OK, full schema verified | PASS |
| TC05 | Citizen position check | 200 OK, returns {position: int} | 200 OK, accurate position | PASS |
| TC06 | Citizen Cancel | 200 OK, status -> CANCELLED, position shifts | 200 OK, `status='CANCELLED'` | PASS |
| TC07 | Post-Cancel Position Shift | Remaining token position decreases by 1 | Token 3 position became 2 | PASS |
| TC08 | Staff Call Next | 200 OK, assigns token 1, status -> CALLED | 200 OK, `status='CALLED'` | PASS |
| TC09 | Staff Start Serving | 200 OK, status -> SERVING | 200 OK, `status='SERVING'` | PASS |
| TC10 | Staff Complete | 200 OK, status -> COMPLETED | 200 OK, `status='COMPLETED'` | PASS |
| TC11 | Staff Skip | 200 OK, status -> SKIPPED | 200 OK, `status='SKIPPED'` | PASS |
| TC12 | Invalid Transition (COMPLETED -> SERVING) | 400 Bad Request, rejected transition | 400 Bad Request, message exact match | PASS |
| TC13 | Invalid Token ID | 404 Not Found | 404 Not Found | PASS |
| TC14 | Invalid Input (String for Integer ID) | 422 Unprocessable Entity | 422 Unprocessable Entity | PASS |

## 5. Citizen Queue Flow
```text
Join Queue (Token 1, 2, 3 generated for RJT-LL with positions 1, 2, 3)
    ↓
ML Wait Prediction (Successfully returned 5.0, 9.0, and 12.0 minutes respectively)
    ↓
Token Retrieved (Positions accurately validated via /position)
    ↓
Cancellation (Token 2 cancelled. Token 3 dynamically shifted to position 2)
```

## 6. Employee Queue Flow
```text
Call Next (Token 1 called)
    ↓
Called (Token 1 status updated to CALLED)
    ↓
Start (Token 1 status updated to SERVING)
    ↓
Serving
    ↓
Complete (Token 1 completed)
    ↓
Call Next (Token 3 called since Token 2 was cancelled)
    ↓
Skip (Token 3 skipped)
```

## 7. ML Integration Verification
- **Prediction Generated:** True.
- **Predicted Wait Persisted:** True (verified via secondary GET lookup).
- **API Response:** `predicted_wait_minutes` is successfully unmarshaled into the JSON response body.
- **Automated Verification:** The existing Phase 3 tests covering graceful degradation and non-finite prediction handling successfully execute within the regression suite. No arbitrary values were mocked during the live manual test.

## 8. Database Verification
- **PostgreSQL Connection:** Solid. All transactions committed successfully.
- **Token Persistence:** Database accurately stores the sequence number (`RJT-LL-001`, `RJT-LL-002`, `RJT-LL-003`).
- **Prediction Persistence:** `predicted_wait_minutes` successfully stored as a column value upon creation.
- **Consistency:** Live API responses match the persistence layer perfectly.

## 9. Error Handling
- **Invalid IDs:** Requesting token `999999` safely returned a `404 Not Found`.
- **Invalid State Transitions:** Attempting to `start` an already `COMPLETED` token safely returned a `400 Bad Request` avoiding data corruption.
- **Validation Errors:** Passing `"abc"` as a `service_id` safely returned a `422 Unprocessable Entity`. 

## 10. Automated Regression Test
```text
python -m pytest -q
```
**Actual Result:**
```text
34 passed
0 failed
0 errors
(24 warnings)
```

## 11. Issues Found
No functional issues found. The backend performed flawlessly during E2E manual execution. 
*(No critical, high, medium, low, or informational bugs discovered).*

## 12. Warnings
- `PydanticDeprecatedSince20`: BaseSettings deprecation.
- `StarletteDeprecationWarning`: TestClient httpx warning.
- `InconsistentVersionWarning`: Scikit-learn model loading (1.9.0 vs 1.9.1).
- `PydanticDeprecatedSince20`: `dict` method deprecated; use `model_dump`.

## 13. Final Verdict
```text
BACKEND MANUAL API TEST: PASS
```
