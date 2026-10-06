import os
import pandas as pd
import numpy as np

def validate_dataset(filepath):
    print(f"--- Validating Test Dataset: {filepath} ---")
    df = pd.read_parquet(filepath)
    
    # 1. Row count
    print(f"1. Row count: {len(df)}")
    
    # 2. Missing values
    missing = df.isnull().sum()
    print("2. Missing values:")
    print(missing[missing > 0])
    
    # 3. Duplicate event IDs
    dup_ids = df['event_id'].duplicated().sum()
    print(f"3. Duplicate event IDs: {dup_ids}")
    
    # 4. Invalid office IDs
    valid_offices = {'RJT', 'AMD', 'STV'}
    invalid_offices = df[~df['office_id'].isin(valid_offices)]
    print(f"4. Invalid office IDs: {len(invalid_offices)}")
    
    # 5. Invalid service IDs
    valid_services = {'LL', 'DL_RENEWAL', 'VEH_REG'}
    invalid_services = df[~df['service_id'].isin(valid_services)]
    print(f"5. Invalid service IDs: {len(invalid_services)}")
    
    # 6. Negative queue values
    neg_queue = df[(df['people_ahead'] < 0) | (df['queue_length'] < 0)]
    print(f"6. Negative queue values: {len(neg_queue)}")
    
    # 7 & 8. Negative wait times & service durations
    completed = df[(df['no_show'] == 0) & (df['cancelled'] == 0)]
    neg_wait = completed[completed['actual_wait_minutes'] < 0]
    neg_service = completed[completed['actual_service_minutes'] <= 0]
    print(f"7. Negative wait times: {len(neg_wait)}")
    print(f"8. Zero/Negative service durations: {len(neg_service)}")
    
    # 9, 10, 11, 12, 13. Timestamp consistency
    start_before_arrival = completed[completed['service_start_time'] < completed['arrival_time']]
    end_before_start = completed[completed['service_end_time'] < completed['service_start_time']]
    
    wait_mismatch = completed[np.abs(completed['actual_wait_minutes'] - (completed['service_start_time'] - completed['arrival_time']).dt.total_seconds() / 60) > 0.01]
    service_mismatch = completed[np.abs(completed['actual_service_minutes'] - (completed['service_end_time'] - completed['service_start_time']).dt.total_seconds() / 60) > 0.01]
    
    print(f"10. service_start_time < arrival_time: {len(start_before_arrival)}")
    print(f"11. service_end_time < service_start_time: {len(end_before_start)}")
    print(f"12. Wait time mismatch: {len(wait_mismatch)}")
    print(f"13. Service time mismatch: {len(service_mismatch)}")
    
    # 14. No-show/Cancel consistency
    no_shows = df[(df['no_show'] == 1) | (df['cancelled'] == 1)]
    bad_no_shows = no_shows[~no_shows['service_start_time'].isnull()]
    print(f"14. Inconsistent no-shows/cancels (have start time): {len(bad_no_shows)}")
    
    # 15 & 16. Distributions
    print("15. Office distribution:")
    print(df['office_id'].value_counts())
    print("16. Service distribution:")
    print(df['service_id'].value_counts())
    
    # 20. Impossible queue states
    imp_queue = df[df['queue_length'] < df['people_ahead']]
    print(f"20. Impossible queue states (length < ahead): {len(imp_queue)}")
    
    print("--- Validation Complete ---")

if __name__ == '__main__':
    validate_dataset(os.path.join('ml', 'data', 'synthetic', 'test', 'queue_events_test.parquet'))
