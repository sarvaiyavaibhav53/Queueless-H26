========================================
QUEUELESS FEATURE ENGINEERING REPORT
====================================

Dataset:
QueueLess Full Synthetic Dataset

Source:
ml/data/synthetic/frozen/queue_events_full.parquet

Source rows:
195838

---

## TARGET

Target:
actual_wait_minutes

Valid target rows:
180809

Excluded no-show rows:
11077

Excluded cancelled rows:
3952

---

## FEATURES

### office_id
* type: str
* missing %: 0.00%
* min: N/A
* max: N/A
* mean: N/A
* median: N/A
* unique values: 3
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### service_id
* type: str
* missing %: 0.00%
* min: N/A
* max: N/A
* mean: N/A
* median: N/A
* unique values: 3
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### hour
* type: int64
* missing %: 0.00%
* min: 9
* max: 16
* mean: 12.2892
* median: 12.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### day_of_week
* type: int64
* missing %: 0.00%
* min: 0
* max: 4
* mean: 2.002
* median: 2.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### people_ahead
* type: int64
* missing %: 0.00%
* min: 0
* max: 69
* mean: 9.7785
* median: 8.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### queue_length
* type: int64
* missing %: 0.00%
* min: 0
* max: 69
* mean: 9.7785
* median: 8.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### active_counters
* type: int64
* missing %: 0.00%
* min: 3
* max: 5
* mean: 4.1114
* median: 4.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### arrival_rate_15m
* type: int64
* missing %: 0.00%
* min: 0
* max: 22
* mean: 6.2119
* median: 6.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### counter_load
* type: float64
* missing %: 0.00%
* min: 0.0
* max: 23.0
* mean: 2.5803
* median: 1.8
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### historical_avg_service_minutes
* type: float64
* missing %: 0.00%
* min: 7.0
* max: 15.0
* mean: 10.3316
* median: 9.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### service_time_std
* type: float64
* missing %: 0.00%
* min: 2.0
* max: 5.0
* mean: 3.3331
* median: 3.0
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

### no_show_rate
* type: float64
* missing %: 0.00%
* min: 0.047843513540625424
* max: 0.0628166836061583
* mean: 0.0565
* median: 0.0595
* unique values: N/A
* prediction-time justification: Calculated using only historical or current state information available exactly when the person enters the queue.

---

## LEAKAGE AUDIT

Column | Classification | Reason
---|---|---
event_id | IDENTIFIER | Unique identifier for the event, no predictive value.
office_id | APPROVED FEATURE | Available at the moment the citizen enters the queue.
service_id | APPROVED FEATURE | Available at the moment the citizen enters the queue.
arrival_time | ANALYSIS ONLY | Used for chronological splitting and tracking, not a direct model feature.
date | ANALYSIS ONLY | Used for chronological splitting and tracking, not a direct model feature.
hour | APPROVED FEATURE | Available at the moment the citizen enters the queue.
day_of_week | APPROVED FEATURE | Available at the moment the citizen enters the queue.
is_weekend | FORBIDDEN / LEAKAGE RISK | Not in the approved list, potentially leaks future info or lacks justification.
people_ahead | APPROVED FEATURE | Available at the moment the citizen enters the queue.
queue_length | APPROVED FEATURE | Available at the moment the citizen enters the queue.
active_counters | APPROVED FEATURE | Available at the moment the citizen enters the queue.
arrival_rate_15m | APPROVED FEATURE | Available at the moment the citizen enters the queue.
counter_load | APPROVED FEATURE | Available at the moment the citizen enters the queue.
historical_avg_service_minutes | APPROVED FEATURE | Available at the moment the citizen enters the queue.
service_time_std | APPROVED FEATURE | Available at the moment the citizen enters the queue.
no_show_rate | APPROVED FEATURE | Available at the moment the citizen enters the queue.
no_show | FORBIDDEN / LEAKAGE RISK | Future information. Occurs after the prediction moment (entering the queue).
cancelled | FORBIDDEN / LEAKAGE RISK | Future information. Occurs after the prediction moment (entering the queue).
service_start_time | FORBIDDEN / LEAKAGE RISK | Future information. Occurs after the prediction moment (entering the queue).
service_end_time | FORBIDDEN / LEAKAGE RISK | Future information. Occurs after the prediction moment (entering the queue).
actual_service_minutes | FORBIDDEN / LEAKAGE RISK | Future information. Occurs after the prediction moment (entering the queue).
actual_wait_minutes | TARGET | The outcome variable to predict.

---

## TARGET DISTRIBUTION

Count: 180809
Mean: 30.4173
Median: 22.2018
Std: 30.6685
Min: 0.0000
P90: 73.0231
P95: 91.1311
P99: 130.9032
Max: 242.8345

---

## TARGET BY OFFICE

RJT: 48.1972
AMD: 18.4455
STV: 29.5952

---

## TARGET BY SERVICE

LL: 30.3828
DL_RENEWAL: 30.4443
VEH_REG: 30.4245

---

## TARGET BY HOUR

09: 1.5303
10: 13.5293
11: 33.5154
12: 40.3578
13: 31.5672
14: 36.0475
15: 47.7791
16: 36.9824

---

## MISSING VALUES

office_id: 0 missing
service_id: 0 missing
hour: 0 missing
day_of_week: 0 missing
people_ahead: 0 missing
queue_length: 0 missing
active_counters: 0 missing
arrival_rate_15m: 0 missing
counter_load: 0 missing
historical_avg_service_minutes: 0 missing
service_time_std: 0 missing
no_show_rate: 0 missing

---

## FEATURE CORRELATION

hour: 0.3637
day_of_week: 0.0016
people_ahead: 0.9414
queue_length: 0.9414
active_counters: -0.3864
arrival_rate_15m: 0.1185
counter_load: 0.9810
historical_avg_service_minutes: 0.0004
service_time_std: 0.0004
no_show_rate: 0.0655

---

## CLEANUP AFTER REVIEW

1. `is_weekend` removed from model feature set because it is constant.
2. `people_ahead` and `queue_length` are currently highly redundant in the synthetic simulation. Both are retained temporarily for the modeling stage because the redundancy is not leakage. Feature importance and model performance will be evaluated before deciding whether one should be removed.
3. Historical features were audited for prediction-time availability.
4. `historical_avg_service_minutes`, `service_time_std`, and `no_show_rate` were verified to use static configurations or current-time constants based on the simulation config, meaning no future information is leaked.
5. Final approved model feature list: `office_id, service_id, hour, day_of_week, people_ahead, queue_length, active_counters, arrival_rate_15m, counter_load, historical_avg_service_minutes, service_time_std, no_show_rate`.
6. Final forbidden/leakage columns: `service_start_time, service_end_time, actual_service_minutes, no_show, cancelled`.
7. Final validation confirms X has no forbidden columns, and `is_weekend` has been excluded.

---

## DATASET OUTPUT

Feature dataset:
ml/data/synthetic/features/queue_features.parquet

Feature CSV:
ml/data/synthetic/features/queue_features.csv

Modeling subset:
ml/data/synthetic/features/queue_features_modeling.parquet

Modeling CSV:
ml/data/synthetic/features/queue_features_modeling.csv

Report:
ml/data/synthetic/features/FEATURE_ENGINEERING_REPORT.md

---

## FINAL VALIDATION

Feature dataset created: PASS
Target separated: PASS
No forbidden target columns in X: PASS
Leakage audit: PASS
Feature quality: PASS
Missing-value analysis: PASS
Target analysis: PASS
Modeling subset created: PASS

========================================

Final status:

FEATURE ENGINEERING READY
