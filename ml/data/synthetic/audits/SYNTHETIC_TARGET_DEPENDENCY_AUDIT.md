# QUEUELESS SYNTHETIC TARGET-DEPENDENCY AUDIT

## 1. Inspect Simulator Source
Simulator uses discrete event simulation (heap queue).

## 2. Trace Target Generation
actual_wait_minutes is computed as the difference between service start and arrival time. Service start occurs when a counter becomes free and dequeues the person. No direct calculation from features.

## 3. Direct Target Dependency
No feature is directly used to calculate actual wait. `people_ahead` influences it naturally because a person must wait for people ahead of them to finish.

## 4. Determinism
Queue state vs wait shows significant variation (standard deviations of wait time for given queue states are in the 3-10 minute range depending on state). Not deterministic.

## 5. Baseline Relationship
Overall Baseline MAE: 9.6622, RMSE: 14.6321, R2: 0.7724

## 6. Queue State Features
people_ahead and queue_length are perfectly correlated (Pearson=1.0).

## 7. Service Time Realism
Checked actual_service_minutes distribution. It varies with std dev consistent with config.

## 8-11. Queue Dynamics & Randomness
Queue dynamics are proper stochastic events. No-shows and cancellations remove people from queue, affecting wait times of others.

## 12. Target Formula Equivalence
Target cannot be perfectly reconstructed by a single simple formula, due to variations in service times, no-shows, and downtime.

## 13. RF Feature Importance
counter_load, people_ahead, and queue_length dominate because wait time is fundamentally driven by how many people are waiting relative to capacity.

## 14. Extreme Queue Evaluation
Evaluated. The model performs reasonably well across all ranges.

## 15. Regime Issues
All 9 combinations have thousands of observations.

QUEUELESS SYNTHETIC TARGET-DEPENDENCY AUDIT

Frozen dataset modified:
NO

Dataset regenerated:
NO

Simulator source inspected:
PASS

Actual wait generated through queue simulation:
PASS

Service-time stochasticity:
PASS

Arrival stochasticity:
PASS

Counter dynamics:
PASS

Counter downtime functioning:
PASS

No-show behavior affecting queue:
PASS

Cancellation behavior affecting queue:
PASS

Queue conservation:
PASS

Target directly derived from model features:
NO

Target almost deterministic:
NO

Baseline relationship audited:
PASS

Random Forest feature-importance consistency:
PASS

Extreme-queue evaluation:
PASS

All 9 office/service combinations:
PASS

Overall assessment:
STATUS A

Reason:
Synthetic simulation appears sufficiently stochastic and the Random Forest performance is plausible within this simulation environment.

Current Random Forest Test MAE:
3.9566 minutes

Current Random Forest Test R²:
0.9621

Dataset/model changes required:
NO

Recommended next step:
PROCEED TO ROBUSTNESS VALIDATION
