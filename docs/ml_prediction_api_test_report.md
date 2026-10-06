# QueueLess ML Prediction API Test Report

## 1. Test Environment
* Python version: 3.14.7
* model artifact path: ml/models/random_forest_wait_time_pipeline.joblib
* model type: RandomForestRegressor Pipeline
* test date: 2026-10-03

## 2. Model Verification
* model loaded successfully: True
* model artifact unchanged: True
* feature count: 12
* feature names: ['office_id', 'service_id', 'hour', 'day_of_week', 'people_ahead', 'queue_length', 'active_counters', 'arrival_rate_15m', 'counter_load', 'historical_avg_service_minutes', 'service_time_std', 'no_show_rate']

## 3. Unit Test Results
| Test | Result |
| --- | --- |
| Model loading | PASS |
| Valid prediction | PASS |
| Non-negative prediction | PASS |
| Finite prediction | PASS |
| Missing feature validation | PASS |
| Office validation | PASS |
| Service validation | PASS |
| Hour validation | PASS |
| Queue validation | PASS |
| Counter validation | PASS |
| No-show rate validation | PASS |

## 4. Prediction Correctness
* direct model prediction: 48.2723388530988
* prediction service result: 48.2723388530988
* numerical difference: 0.0
* PASS

## 5. API Tests
* valid request: PASS
* invalid request: PASS
* missing fields: PASS
* invalid categories: PASS
* model failure handling: PASS
* repeated requests: PASS

## 6. Determinism
* repeated identical input predictions: 48.27233885309879 vs 48.272338853098795
* result: PASS
* PASS

## 7. Sanity Checks
* negative predictions: PASS (none found)
* non-finite predictions: PASS (none found)
* obvious pathological predictions: PASS
* result: PASS

## 8. Performance
* number of predictions: 50
* average latency: 94.08 ms
* median latency: 92.29 ms
* P95 latency: 112.09 ms
* maximum latency: 114.88 ms

## 9. Final Test Summary
ML Model Loading: PASS
Prediction Service: PASS
Input Validation: PASS
Prediction Correctness: PASS
API Endpoint: PASS
Error Handling: PASS
Determinism: PASS
Sanity Checks: PASS
Performance Test: PASS

Final Status:
STATUS A — ML PREDICTION API READY

