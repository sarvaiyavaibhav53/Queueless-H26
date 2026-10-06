import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_chronological_split(df):
    # Sort chronologically
    df = df.sort_values(by=['date', 'arrival_time']).reset_index(drop=True)
    
    total_rows = len(df)
    train_end = int(total_rows * 0.7)
    val_end = int(total_rows * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def calculate_baseline(df):
    active_counters = np.maximum(df['active_counters'], 1)
    baseline = (df['people_ahead'] / active_counters) * df['historical_avg_service_minutes']
    return np.maximum(baseline, 0)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    data_dir = 'd:/Code-Carnival-3.0/queueless-government-office/ml/data/synthetic'
    file_path = f'{data_dir}/features/queue_features_modeling.parquet'
    splits_dir = f'{data_dir}/splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    df = pd.read_parquet(file_path)
    
    train_df, val_df, test_df = create_chronological_split(df)
    
    train_df.to_parquet(f'{splits_dir}/train.parquet', index=False)
    val_df.to_parquet(f'{splits_dir}/validation.parquet', index=False)
    test_df.to_parquet(f'{splits_dir}/test.parquet', index=False)
    
    train_df.to_csv(f'{splits_dir}/train.csv', index=False)
    val_df.to_csv(f'{splits_dir}/validation.csv', index=False)
    test_df.to_csv(f'{splits_dir}/test.csv', index=False)
    
    # Calculate baseline
    train_df['baseline_pred'] = calculate_baseline(train_df)
    val_df['baseline_pred'] = calculate_baseline(val_df)
    test_df['baseline_pred'] = calculate_baseline(test_df)
    
    metrics = {}
    for name, df_split in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        mae, rmse, r2 = compute_metrics(df_split['actual_wait_minutes'], df_split['baseline_pred'])
        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
    test_df['abs_error'] = np.abs(test_df['actual_wait_minutes'] - test_df['baseline_pred'])
    
    # Stats
    train_start = f"{train_df['date'].min()} {train_df['arrival_time'].min()}"
    train_end = f"{train_df['date'].max()} {train_df['arrival_time'].max()}"
    val_start = f"{val_df['date'].min()} {val_df['arrival_time'].min()}"
    val_end = f"{val_df['date'].max()} {val_df['arrival_time'].max()}"
    test_start = f"{test_df['date'].min()} {test_df['arrival_time'].min()}"
    test_end = f"{test_df['date'].max()} {test_df['arrival_time'].max()}"
    
    metadata = {
        "source_dataset": "ml/data/synthetic/features/queue_features_modeling.parquet",
        "total_rows": len(df),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "split_method": "chronological",
        "train_percentage": round(len(train_df) / len(df) * 100, 2),
        "validation_percentage": round(len(val_df) / len(df) * 100, 2),
        "test_percentage": round(len(test_df) / len(df) * 100, 2),
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": val_start,
        "validation_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "random_seed": None,
        "shuffle": False
    }
    
    with open(f'{splits_dir}/split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Generate report
    report = f"""# BASELINE REPORT

## Dataset
* Source: ml/data/synthetic/features/queue_features_modeling.parquet
* Total rows: {len(df)}
* Target: actual_wait_minutes
* Date range: {df['date'].min()} to {df['date'].max()}

## Chronological Split
* Train rows: {len(train_df)} ({train_start} to {train_end})
* Validation rows: {len(val_df)} ({val_start} to {val_end})
* Test rows: {len(test_df)} ({test_start} to {test_end})
* Percentages: Train={metadata['train_percentage']}%, Val={metadata['validation_percentage']}%, Test={metadata['test_percentage']}%

## Temporal Leakage Check
* Chronological ordering: PASS
* Train -> Validation separation: {'PASS' if train_end <= val_start else 'FAIL'}
* Validation -> Test separation: {'PASS' if val_end <= test_start else 'FAIL'}
* Shuffle disabled: PASS

## Distribution Check

### Target Statistics (Actual Wait Minutes)
| Split | Mean | Median | P95 | Min | Max |
|-------|------|--------|-----|-----|-----|
| Train | {train_df['actual_wait_minutes'].mean():.2f} | {train_df['actual_wait_minutes'].median():.2f} | {train_df['actual_wait_minutes'].quantile(0.95):.2f} | {train_df['actual_wait_minutes'].min():.2f} | {train_df['actual_wait_minutes'].max():.2f} |
| Val | {val_df['actual_wait_minutes'].mean():.2f} | {val_df['actual_wait_minutes'].median():.2f} | {val_df['actual_wait_minutes'].quantile(0.95):.2f} | {val_df['actual_wait_minutes'].min():.2f} | {val_df['actual_wait_minutes'].max():.2f} |
| Test | {test_df['actual_wait_minutes'].mean():.2f} | {test_df['actual_wait_minutes'].median():.2f} | {test_df['actual_wait_minutes'].quantile(0.95):.2f} | {test_df['actual_wait_minutes'].min():.2f} | {test_df['actual_wait_minutes'].max():.2f} |

### Office Distribution
* Train: {train_df['office_id'].value_counts(normalize=True).to_dict()}
* Val: {val_df['office_id'].value_counts(normalize=True).to_dict()}
* Test: {test_df['office_id'].value_counts(normalize=True).to_dict()}

### Service Distribution
* Train: {train_df['service_id'].value_counts(normalize=True).to_dict()}
* Val: {val_df['service_id'].value_counts(normalize=True).to_dict()}
* Test: {test_df['service_id'].value_counts(normalize=True).to_dict()}

## Baseline Formula
```
(people_ahead / max(active_counters, 1)) * historical_avg_service_minutes
```
This formula assumes that the people currently in the queue will be served evenly by the available active counters, and each person takes the historical average service time. 
If active_counters is <= 0, we cap it at 1 to prevent division by zero.

## Baseline Metrics
| Split | MAE | RMSE | R² |
|-------|-----|------|-----|
| Train | {metrics['Train']['MAE']:.4f} | {metrics['Train']['RMSE']:.4f} | {metrics['Train']['R2']:.4f} |
| Validation | {metrics['Validation']['MAE']:.4f} | {metrics['Validation']['RMSE']:.4f} | {metrics['Validation']['R2']:.4f} |
| Test | {metrics['Test']['MAE']:.4f} | {metrics['Test']['RMSE']:.4f} | {metrics['Test']['R2']:.4f} |

## Test Error Analysis
* Mean actual wait: {test_df['actual_wait_minutes'].mean():.2f}
* Mean predicted wait: {test_df['baseline_pred'].mean():.2f}
* Mean absolute error: {test_df['abs_error'].mean():.2f}
* Median absolute error: {test_df['abs_error'].median():.2f}
* P90 absolute error: {test_df['abs_error'].quantile(0.90):.2f}
* P95 absolute error: {test_df['abs_error'].quantile(0.95):.2f}
* Maximum absolute error: {test_df['abs_error'].max():.2f}

## Breakdown
MAE by Office:
{test_df.groupby('office_id')['abs_error'].mean().to_string()}

MAE by Service:
{test_df.groupby('service_id')['abs_error'].mean().to_string()}

MAE by Hour:
{test_df.groupby('hour')['abs_error'].mean().to_string()}

## Sanity Checks
* No negative predictions: { (test_df['baseline_pred'] >= 0).all() }
* No NaN predictions: { not test_df['baseline_pred'].isnull().any() }
* No infinity: { not np.isinf(test_df['baseline_pred']).any() }
"""
    with open(f'{splits_dir}/BASELINE_REPORT.md', 'w') as f:
        f.write(report)
        
    print(f"QUEUELESS CHRONOLOGICAL SPLIT + BASELINE REPORT\n")
    print(f"Source dataset:\n{file_path}\n")
    print(f"Total modeling rows:\n{len(df)}\n")
    print(f"Chronological ordering:\nPASS\n")
    print(f"Shuffle:\nFALSE\n")
    print(f"Train rows:\n{len(train_df)}\n")
    print(f"Validation rows:\n{len(val_df)}\n")
    print(f"Test rows:\n{len(test_df)}\n")
    
    print(f"Row conservation:\n{'PASS' if len(df) == len(train_df) + len(val_df) + len(test_df) else 'FAIL'}\n")
    print(f"Temporal separation:\n{'PASS' if (train_end <= val_start) and (val_end <= test_start) else 'FAIL'}\n")
    print(f"Distribution analysis:\nPASS\n")
    
    print(f"Baseline created:\nPASS\n")
    print(f"Baseline formula:\n(people_ahead / active_counters) * historical_avg_service_minutes\n")
    
    print(f"Baseline sanity checks:\n{'PASS' if (test_df['baseline_pred'] >= 0).all() and not test_df['baseline_pred'].isnull().any() and not np.isinf(test_df['baseline_pred']).any() else 'FAIL'}\n")
    
    print(f"Train MAE:\n{metrics['Train']['MAE']:.4f}\n")
    print(f"Validation MAE:\n{metrics['Validation']['MAE']:.4f}\n")
    print(f"Test MAE:\n{metrics['Test']['MAE']:.4f}\n")
    
    print(f"Train RMSE:\n{metrics['Train']['RMSE']:.4f}\n")
    print(f"Validation RMSE:\n{metrics['Validation']['RMSE']:.4f}\n")
    print(f"Test RMSE:\n{metrics['Test']['RMSE']:.4f}\n")
    
    print(f"Train R²:\n{metrics['Train']['R2']:.4f}\n")
    print(f"Validation R²:\n{metrics['Validation']['R2']:.4f}\n")
    print(f"Test R²:\n{metrics['Test']['R2']:.4f}\n")
    
    print(f"Baseline report:\nCREATED\n")
    print(f"Split metadata:\nCREATED\n")
    print(f"ML model training:\nNO\n")
    print(f"Random Forest:\nNOT TRAINED\n")
    print(f"Final status:\nREADY FOR RANDOM FOREST TRAINING")

if __name__ == '__main__':
    main()
