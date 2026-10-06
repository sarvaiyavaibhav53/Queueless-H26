import heapq
import random
import pandas as pd
import numpy as np
from service_process import ServiceProcess
from queue_engine import QueueEngine
from counter_manager import CounterManager

class EventEngine:
    def __init__(self, config):
        self.config = config
        self.service_process = ServiceProcess(config)
        self.queue_engine = QueueEngine(config['offices'].keys())
        self.counter_manager = CounterManager(config)
        
        self.events = []
        self.event_counter = 0
        self.completed_events = []
        self.recent_arrivals = {o: [] for o in config['offices']}
        
    def add_event(self, timestamp, event_type, data):
        self.event_counter += 1
        heapq.heappush(self.events, (timestamp, self.event_counter, event_type, data))
        
    def run_day(self, date, arrivals):
        self.events = []
        self.queue_engine = QueueEngine(self.config['offices'].keys())
        self.counter_manager = CounterManager(self.config)
        self.recent_arrivals = {o: [] for o in self.config['offices']}
        
        for arr in arrivals:
            self.add_event(arr['arrival_time'], 'ARRIVAL', arr)
            
        while self.events:
            ts, _, e_type, data = heapq.heappop(self.events)
            
            if e_type == 'ARRIVAL':
                self.handle_arrival(ts, data)
            elif e_type == 'SERVICE_END':
                self.handle_service_end(ts, data)
            elif e_type == 'COUNTER_BACK':
                self.handle_counter_back(ts, data)
                
    def handle_arrival(self, ts, person):
        office_id = person['office_id']
        service_id = person['service_id']
        
        self.recent_arrivals[office_id] = [t for t in self.recent_arrivals[office_id] if (ts - t).total_seconds() <= 900]
        arrival_rate_15m = len(self.recent_arrivals[office_id])
        self.recent_arrivals[office_id].append(ts)
        
        people_ahead = self.queue_engine.length(office_id)
        total_counters = self.counter_manager.get_total(office_id)
        
        base_no_show = self.config['no_show']['base_rate']
        daily_var_ns = np.random.normal(1.0, self.config['no_show']['daily_variation_std'])
        is_peak = self.config['no_show']['peak_hour_multiplier'] if ts.hour in [10, 11, 14, 15] else 1.0
        no_show_prob = base_no_show * daily_var_ns * is_peak
        
        is_cancelled = False
        is_no_show = False
        r = random.random()
        if r < self.config['cancellation']['base_rate']:
            is_cancelled = True
        elif r < self.config['cancellation']['base_rate'] + no_show_prob:
            is_no_show = True
        
        person.update({
            'date': ts.strftime('%Y-%m-%d'),
            'hour': ts.hour,
            'day_of_week': ts.dayofweek,
            'is_weekend': int(ts.dayofweek >= 5),
            'people_ahead': people_ahead,
            'queue_length': people_ahead,
            'active_counters': total_counters,
            'arrival_rate_15m': arrival_rate_15m,
            'counter_load': people_ahead / max(1, total_counters),
            'historical_avg_service_minutes': self.config['services'][service_id]['mean_minutes'],
            'service_time_std': self.config['services'][service_id]['std_minutes'],
            'no_show_rate': no_show_prob,
            'no_show': int(is_no_show),
            'cancelled': int(is_cancelled)
        })
        
        if is_no_show or is_cancelled:
            person['service_start_time'] = pd.NaT
            person['service_end_time'] = pd.NaT
            person['actual_service_minutes'] = np.nan
            person['actual_wait_minutes'] = np.nan
            self.completed_events.append(person)
            return
            
        self.queue_engine.enqueue(office_id, person)
        self.try_assign_counter(ts, office_id)
        
    def handle_service_end(self, ts, data):
        person = data['person']
        person['service_end_time'] = ts
        person['actual_service_minutes'] = (ts - person['service_start_time']).total_seconds() / 60.0
        self.completed_events.append(person)
        
        office_id = person['office_id']
        
        if random.random() < self.config['counter_downtime']['probability_per_service']:
            min_d = self.config['counter_downtime']['duration_range']['min_minutes']
            max_d = self.config['counter_downtime']['duration_range']['max_minutes']
            downtime = random.uniform(min_d, max_d)
            self.add_event(ts + pd.Timedelta(minutes=downtime), 'COUNTER_BACK', {'office_id': office_id})
        else:
            self.counter_manager.release(office_id)
            self.try_assign_counter(ts, office_id)
            
    def handle_counter_back(self, ts, data):
        office_id = data['office_id']
        self.counter_manager.release(office_id)
        self.try_assign_counter(ts, office_id)
        
    def try_assign_counter(self, ts, office_id):
        if self.counter_manager.is_free(office_id) and self.queue_engine.length(office_id) > 0:
            person = self.queue_engine.dequeue(office_id)
            self.counter_manager.allocate(office_id)
            
            person['service_start_time'] = ts
            person['actual_wait_minutes'] = (ts - person['arrival_time']).total_seconds() / 60.0
            
            duration = self.service_process.sample_duration(person['service_id'])
            end_time = ts + pd.Timedelta(minutes=duration)
            
            self.add_event(end_time, 'SERVICE_END', {'person': person})
