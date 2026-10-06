# RANDOM FOREST REPORT

## Dataset
* Source: ml/data/synthetic/splits/
* Train rows: 126566
* Validation rows: 27121
* Test rows: 27122

## Features
['office_id', 'service_id', 'hour', 'day_of_week', 'people_ahead', 'queue_length', 'active_counters', 'arrival_rate_15m', 'counter_load', 'historical_avg_service_minutes', 'service_time_std', 'no_show_rate']

## Preprocessing
* office_id one-hot encoding
* service_id one-hot encoding
* no scaling
* encoder fitted only on training data (refitted on train+val for final model)

## Hyperparameter Search
* Params: {'n_estimators': 300, 'max_depth': 25, 'min_samples_leaf': 2, 'max_features': 1.0} => MAE: 4.3533, RMSE: 6.4668, R2: 0.9595
* Params: {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 4, 'max_features': 'sqrt'} => MAE: 4.2191, RMSE: 6.2749, R2: 0.9618
* Params: {'n_estimators': 500, 'max_depth': 25, 'min_samples_leaf': 2, 'max_features': 1.0} => MAE: 4.3489, RMSE: 6.4611, R2: 0.9595

## Selected Model
* hyperparameters: {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 4, 'max_features': 'sqrt'}
* validation MAE: 4.2191
* validation RMSE: 6.2749
* validation R²: 0.9618

## Final Test Results
* MAE: 3.9566
* RMSE: 5.9948
* R²: 0.9621

## Baseline vs Random Forest
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline | 9.4442 | 14.5253 | 0.7772 |
| Random Forest | 3.9566 | 5.9948 | 0.9621 |

## Improvement
* absolute MAE improvement: 5.4876
* percentage MAE improvement: 58.11%
* absolute RMSE improvement: 8.5305
* percentage RMSE improvement: 58.73%

## Feature Importance
                       Feature  Importance
                  counter_load    0.388589
                  queue_length    0.249462
                  people_ahead    0.246755
                          hour    0.033741
               active_counters    0.030240
                 office_id_RJT    0.023082
                 office_id_AMD    0.010041
                  no_show_rate    0.006148
              arrival_rate_15m    0.005384
                 office_id_STV    0.002892
                   day_of_week    0.002681
historical_avg_service_minutes    0.000254
              service_time_std    0.000254
         service_id_DL_RENEWAL    0.000186
                 service_id_LL    0.000147
            service_id_VEH_REG    0.000143

## Error Analysis
* Median absolute error: 2.49
* P90 absolute error: 9.60
* P95 absolute error: 13.01
* Maximum absolute error: 53.71

### MAE by office
office_id
AMD    2.687468
RJT    5.973345
STV    3.727978

### MAE by service
service_id
DL_RENEWAL    3.846031
LL            4.010530
VEH_REG       4.014439

### MAE by hour
hour
9     0.903052
10    2.796475
11    4.283063
12    4.638070
13    4.042353
14    4.539468
15    5.593190
16    4.099726

## Overfitting Check
| Split | MAE | RMSE | R² |
|-------|-----|------|-----|
| Train (Refitted model) | 3.1773 | 4.7449 | 0.9755 |
| Validation (Refitted model) | 3.3085 | 4.9302 | 0.9764 |
| Test (Final unseen evaluation) | 3.9566 | 5.9948 | 0.9621 |

## Reproducibility
* random seed: 42
* split source: ml/data/synthetic/splits/
* model configuration: {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 4, 'max_features': 'sqrt'}
* preprocessing: OneHotEncoder
* model file: ml/models/random_forest_wait_time_pipeline.joblib
* training script: ml/src/train_random_forest.py
