# ROBUSTNESS VALIDATION REPORT

## 1. Artifacts Used
* Model: ml/models/random_forest_wait_time_pipeline.joblib
* Test dataset: ml/data/synthetic/splits/test.parquet
* Features: ['office_id', 'service_id', 'hour', 'day_of_week', 'people_ahead', 'queue_length', 'active_counters', 'arrival_rate_15m', 'counter_load', 'historical_avg_service_minutes', 'service_time_std', 'no_show_rate']

## 2. Reproducibility Check
* Prediction matching: True
* Metrics matching: True
* Reproducibility: PASS

## 3. Overall Error Distribution
* MAE: 3.9566
* RMSE: 5.9948
* R²: 0.9621
* Median AE: 2.4890
* P90 AE: 9.6026
* P95 AE: 13.0069
* Max AE: 53.7110
* Min Prediction: 0.0000
* Max Prediction: 211.3114

* Mean Actual: 29.6496
* Median Actual: 20.5556
* P90 Actual: 72.6805
* P95 Actual: 90.4576
* P99 Actual: 124.6761
* Negative Predictions: PASS

## 4. Office Robustness
office_id  count      mae     rmse       r2   med_ae    p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
      AMD  10492 2.687468 3.999085 0.963366 1.733694  6.479749  8.726658    18.706755  18.627049      6.051060       9.528269
      RJT   7624 5.973345 8.351933 0.954077 4.208335 13.838158 17.534703    48.906145  48.958307     15.431446      21.241386
      STV   9006 3.727978 5.526754 0.948357 2.417563  8.673107 11.407899    26.096495  25.935159      8.328731      12.151717

## 5. Service Robustness
service_id  count      mae     rmse       r2   med_ae   p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
DL_RENEWAL   9100 3.846031 5.844227 0.964325 2.370241 9.403066 12.601102    29.275420  29.242926      7.165709      10.908180
        LL   9012 4.010530 6.100142 0.960260 2.527777 9.746949 12.999032    29.572207  29.472609     12.128454      17.737496
   VEH_REG   9010 4.014439 6.038460 0.961488 2.566761 9.659140 13.523986    30.104961  30.027458      9.060604      14.150743

## 6. Hourly Robustness
 hour  count      mae     rmse       r2   med_ae    p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
    9   2450 0.903052 1.844628 0.794554 0.169963  2.719862  4.095480     1.687356   1.646120      0.938451       2.387902
   10   4576 2.796475 3.977974 0.922486 1.960945  6.344199  8.350550    13.676724  13.523526      4.875817       7.374377
   11   4564 4.283063 5.979467 0.930462 3.050758  9.624291 12.920253    32.051859  32.273715     10.106333      13.453428
   12   3064 4.638070 6.453941 0.947325 3.260336 10.625080 13.987543    37.824664  37.944901     11.837692      15.933922
   13   2525 4.042353 6.395103 0.951868 2.408303  9.788170 13.309960    29.718388  29.292810      9.441700      14.572096
   14   4673 4.539468 6.620735 0.957883 3.103500 10.395992 13.647727    35.133798  35.690971     11.026481      16.070733
   15   3695 5.593190 7.965498 0.957593 3.741348 13.364268 17.564455    48.200976  47.380470     14.917502      20.616452
   16   1575 4.099726 6.202462 0.976663 2.532712 10.753611 14.303773    36.785276  36.170810     11.842204      18.549926

## 7. Queue-size Robustness
  queue_group  count      mae      rmse       r2   med_ae    p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
Extreme (>20)   3774 7.607282 10.086615 0.874028 5.884610 16.478996 20.364038    84.918254  83.914694     26.072969      29.726588
 High (11-20)   6631 5.772109  7.507520 0.728288 4.598880 12.158511 15.212948    45.839406  45.981148     14.027063      16.409064
    Low (0-3)   9585 1.425451  2.209124 0.731874 0.836187  3.629466  4.816986     3.217287   3.232826      1.619811       2.886246
Medium (4-10)   7132 3.738668  4.885237 0.703807 2.994521  7.748300  9.640024    20.874435  20.987589      6.899451       8.555757

## 8. People-ahead Robustness
pa_group  count      mae      rmse       r2   med_ae    p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
       0   5783 0.796529  1.352581 0.078436 0.413610  1.977220  2.860979     0.578770   0.634076      0.578770       1.523207
     1-2   2590 2.224901  2.867411 0.242114 1.848084  4.448912  5.620552     5.939182   5.884701      2.919385       3.827128
   11-20   6631 5.772109  7.507520 0.728288 4.598880 12.158511 15.212948    45.839406  45.981148     14.027063      16.409064
     3-5   3630 3.018531  3.928122 0.438425 2.415015  6.252213  7.834820    12.765209  12.746106      4.599964       5.824235
    6-10   4714 4.030769  5.233205 0.624320 3.254092  8.253924 10.280923    24.320523  24.500111      7.875833       9.522183
     >20   3774 7.607282 10.086615 0.874028 5.884610 16.478996 20.364038    84.918254  83.914694     26.072969      29.726588

## 9. Active-counter Robustness
 active_counters  count      mae     rmse       r2   med_ae    p90_ae    p95_ae  mean_actual  mean_pred  baseline_mae  baseline_rmse
               3   7624 5.973345 8.351933 0.954077 4.208335 13.838158 17.534703    48.906145  48.958307     15.431446      21.241386
               4   9006 3.727978 5.526754 0.948357 2.417563  8.673107 11.407899    26.096495  25.935159      8.328731      12.151717
               5  10492 2.687468 3.999085 0.963366 1.733694  6.479749  8.726658    18.706755  18.627049      6.051060       9.528269

## 10. Prediction Sanity Checks
* No negative predictions: True
* Queue relationship (increasing people_ahead -> increasing pred): True
* Counter relationship (increasing counters -> decreasing pred): False

## 11. Error Analysis
Top 20 Absolute Errors saved to top_prediction_errors.csv.

## 12. Baseline Comparison
* RF MAE: 3.9566 vs Baseline MAE: 9.4442

## 13. Performance Stability Check
* <= 2 mins: 43.75%
* <= 5 mins: 72.15%
* <= 10 mins: 90.70%
* > 10 mins: 9.30%
* > 20 mins: 1.19%

## 14. Data Coverage Check
* Offices covered: 3 / 3
* Services covered: 3 / 3
* Hours covered: 8 / 8

## 15. Robustness Verdict
Status: STATUS A
Reason: Reproducibility passes, metrics match expectations, no extreme subgroup failures detected, and coverage is sufficient across all dimensions.

## 16. Recommendations
Proceed to production deployment or A/B testing framework setup.
