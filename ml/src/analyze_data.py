import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(filepath, out_dir):
    print("Loading test data for analysis...")
    df = pd.read_parquet(filepath)
    completed = df[(df['no_show'] == 0) & (df['cancelled'] == 0)]
    
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Wait-time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(completed['actual_wait_minutes'], bins=50, kde=True)
    plt.title('Wait Time Distribution')
    plt.xlabel('Wait Time (Minutes)')
    plt.savefig(os.path.join(out_dir, 'wait_time_dist.png'))
    plt.close()
    
    # 2. Service-time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(completed['actual_service_minutes'], bins=50, kde=True)
    plt.title('Service Time Distribution')
    plt.xlabel('Service Time (Minutes)')
    plt.savefig(os.path.join(out_dir, 'service_time_dist.png'))
    plt.close()
    
    # 3. Queue-length over time
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('arrival_time')
    sns.lineplot(data=df_sorted, x='hour', y='queue_length', hue='office_id')
    plt.title('Queue Length at Arrival by Hour')
    plt.savefig(os.path.join(out_dir, 'queue_length_time.png'))
    plt.close()
    
    # 4. Average wait by office
    plt.figure(figsize=(8, 5))
    sns.barplot(data=completed, x='office_id', y='actual_wait_minutes')
    plt.title('Average Wait Time by Office')
    plt.savefig(os.path.join(out_dir, 'avg_wait_by_office.png'))
    plt.close()
    
    # 5. Average wait by service
    plt.figure(figsize=(8, 5))
    sns.barplot(data=completed, x='service_id', y='actual_wait_minutes')
    plt.title('Average Wait Time by Service')
    plt.savefig(os.path.join(out_dir, 'avg_wait_by_service.png'))
    plt.close()
    
    # 6. Average wait by hour
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=completed, x='hour', y='actual_wait_minutes', marker='o')
    plt.title('Average Wait Time by Hour of Day')
    plt.savefig(os.path.join(out_dir, 'avg_wait_by_hour.png'))
    plt.close()
    
    # 7. Arrival rate by hour
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='hour', color='skyblue')
    plt.title('Arrivals by Hour')
    plt.savefig(os.path.join(out_dir, 'arrivals_by_hour.png'))
    plt.close()
    
    # Summary
    no_show_rate = df['no_show'].mean() * 100
    cancellation_rate = df['cancelled'].mean() * 100
    p95_wait = completed['actual_wait_minutes'].quantile(0.95) if not completed.empty else 0
    max_wait = completed['actual_wait_minutes'].max() if not completed.empty else 0
    
    overlap = df[(df['no_show'] == 1) & (df['cancelled'] == 1)]
    
    summary = (f"--- Test Analysis Summary ---\n"
               f"Total events: {len(df)}\n"
               f"Events by office: \n{df['office_id'].value_counts().to_string()}\n"
               f"Events by service: \n{df['service_id'].value_counts().to_string()}\n"
               f"Events by day: \n{df['date'].value_counts().sort_index().to_string()}\n\n"
               f"Expected total: 2700\n"
               f"Actual total: {len(df)}\n"
               f"Difference: {len(df) - 2700}\n"
               f"Difference percentage: {((len(df) - 2700) / 2700) * 100:.2f}%\n\n"
               f"Completed: {len(completed)}\n"
               f"No-show: {len(df[df['no_show'] == 1])}\n"
               f"Cancelled: {len(df[df['cancelled'] == 1])}\n"
               f"No-show + cancelled: {len(overlap)}\n\n"
               f"Average wait: {completed['actual_wait_minutes'].mean():.2f} min\n"
               f"Median wait: {completed['actual_wait_minutes'].median():.2f} min\n"
               f"P95 wait: {p95_wait:.2f} min\n"
               f"Maximum wait: {max_wait:.2f} min\n"
               f"Average queue length: {df['queue_length'].mean():.2f}\n"
               f"Maximum queue length: {df['queue_length'].max()}\n\n"
               f"Average service duration: {completed['actual_service_minutes'].mean():.2f} min\n"
               f"LL average: {completed[completed['service_id'] == 'LL']['actual_service_minutes'].mean():.2f} min\n"
               f"DL_RENEWAL average: {completed[completed['service_id'] == 'DL_RENEWAL']['actual_service_minutes'].mean():.2f} min\n"
               f"VEH_REG average: {completed[completed['service_id'] == 'VEH_REG']['actual_service_minutes'].mean():.2f} min\n")
               
    with open(os.path.join(out_dir, 'summary_statistics_test.txt'), 'w') as f:
        f.write(summary)
        
    print(summary)
    print(f"Analysis complete. Plots saved to {out_dir}")

if __name__ == '__main__':
    in_path = os.path.join('ml', 'data', 'synthetic', 'test', 'queue_events_test.parquet')
    out_path = os.path.join('ml', 'data', 'synthetic', 'test', 'analysis')
    analyze_data(in_path, out_path)
