import os
import yaml
import random
import numpy as np
import pandas as pd
from arrival_process import ArrivalProcess
from event_engine import EventEngine

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'simulation_config.yaml')
    config = load_config(config_path)
    
    np.random.seed(config['simulation']['random_seed'])
    random.seed(config['simulation']['random_seed'])
    
    start_date = pd.Timestamp('2023-01-01')
    days = config['simulation']['test']['days_to_simulate']
    
    arr_process = ArrivalProcess(config)
    sim = EventEngine(config)
    
    days_simulated = 0
    day_offset = 0
    while days_simulated < days:
        current_date = start_date + pd.Timedelta(days=day_offset)
        day_offset += 1
        if current_date.dayofweek >= 5: continue
        
        arrivals = arr_process.generate_daily_arrivals(current_date)
        sim.run_day(current_date, arrivals)
        days_simulated += 1
        
    df = pd.DataFrame(sim.completed_events)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'test')
    os.makedirs(out_dir, exist_ok=True)
    
    df.to_csv(os.path.join(out_dir, 'queue_events_test.csv'), index=False)
    df.to_parquet(os.path.join(out_dir, 'queue_events_test.parquet'), index=False)
    print(f"Test simulation complete: {len(df)} events generated over {days} days.")

if __name__ == '__main__':
    main()
