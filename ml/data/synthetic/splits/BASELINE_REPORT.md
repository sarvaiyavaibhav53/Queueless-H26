# BASELINE REPORT

## Dataset
* Source: ml/data/synthetic/features/queue_features_modeling.parquet
* Total rows: 180809
* Target: actual_wait_minutes
* Date range: 2023-01-02 to 2024-05-24

## Chronological Split
* Train rows: 126566 (2023-01-02 2023-01-02 09:01:47 to 2023-12-25 2023-12-25 15:02:56)
* Validation rows: 27121 (2023-12-25 2023-12-25 15:03:25 to 2024-03-08 2024-03-08 15:13:58)
* Test rows: 27122 (2024-03-08 2024-03-08 15:14:00 to 2024-05-24 2024-05-24 16:58:25)
* Percentages: Train=70.0%, Val=15.0%, Test=15.0%

## Temporal Leakage Check
* Chronological ordering: PASS
* Train -> Validation separation: PASS
* Validation -> Test separation: PASS
* Shuffle disabled: PASS

## Distribution Check

### Target Statistics (Actual Wait Minutes)
| Split | Mean | Median | P95 | Min | Max |
|-------|------|--------|-----|-----|-----|
| Train | 29.98 | 22.03 | 90.30 | 0.00 | 228.24 |
| Val | 33.20 | 24.62 | 95.40 | 0.00 | 214.25 |
| Test | 29.65 | 20.56 | 90.46 | 0.00 | 242.83 |

### Office Distribution
* Train: {'AMD': 0.390223282714157, 'STV': 0.33347028427855824, 'RJT': 0.27630643300728475}
* Val: {'AMD': 0.38169684008701743, 'STV': 0.342059658567162, 'RJT': 0.2762435013458206}
* Test: {'AMD': 0.3868446279772878, 'STV': 0.33205515817417597, 'RJT': 0.28110021384853623}

### Service Distribution
* Train: {'DL_RENEWAL': 0.3340944645481409, 'VEH_REG': 0.3334544822464169, 'LL': 0.3324510532054422}
* Val: {'DL_RENEWAL': 0.33988422255816525, 'LL': 0.3307400169610265, 'VEH_REG': 0.32937576048080824}
* Test: {'DL_RENEWAL': 0.33552097927881425, 'LL': 0.33227638079787625, 'VEH_REG': 0.3322026399233095}

## Baseline Formula
```
(people_ahead / max(active_counters, 1)) * historical_avg_service_minutes
```
This formula assumes that the people currently in the queue will be served evenly by the available active counters, and each person takes the historical average service time. 
If active_counters is <= 0, we cap it at 1 to prevent division by zero.

## Baseline Metrics
| Split | MAE | RMSE | R² |
|-------|-----|------|-----|
| Train | 9.5305 | 14.4547 | 0.7723 |
| Validation | 10.4947 | 15.5332 | 0.7662 |
| Test | 9.4442 | 14.5253 | 0.7772 |

## Test Error Analysis
* Mean actual wait: 29.65
* Mean predicted wait: 25.97
* Mean absolute error: 9.44
* Median absolute error: 5.79
* P90 absolute error: 24.11
* P95 absolute error: 31.61
* Maximum absolute error: 108.04

## Breakdown
MAE by Office:
office_id
AMD     6.051060
RJT    15.431446
STV     8.328731

MAE by Service:
service_id
DL_RENEWAL     7.165709
LL            12.128454
VEH_REG        9.060604

MAE by Hour:
hour
9      0.938451
10     4.875817
11    10.106333
12    11.837692
13     9.441700
14    11.026481
15    14.917502
16    11.842204

## Sanity Checks
* No negative predictions: True
* No NaN predictions: True
* No infinity: True
