import numpy as np
import random
import pandas as pd
import uuid

class ArrivalProcess:
    def __init__(self, config):
        self.config = config

    def generate_daily_arrivals(self, date):
        arrivals = []
        base_vol = self.config['arrivals']['base_daily_citizens_per_service']
        
        weights_dict = self.config['arrivals']['hourly_weights']
        hours = list(weights_dict.keys())
        weights = list(weights_dict.values())
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        for office_id, off_cfg in self.config['offices'].items():
            mult = off_cfg['demand_multiplier']
            expected_daily_arrivals = base_vol * mult
            
            for service_id in self.config['services'].keys():
                daily_var = np.random.normal(1.0, self.config['arrivals']['daily_variation_std'])
                daily_vol = max(1, int(expected_daily_arrivals * daily_var))
                
                hr_counts = np.random.multinomial(daily_vol, probs)
                
                for hr, count in zip(hours, hr_counts):
                    for _ in range(count):
                        minute = random.randint(0, 59)
                        second = random.randint(0, 59)
                        arr_time = pd.Timestamp(date) + pd.Timedelta(hours=hr, minutes=minute, seconds=second)
                        
                        person = {
                            'event_id': str(uuid.uuid4()),
                            'office_id': office_id,
                            'service_id': service_id,
                            'arrival_time': arr_time
                        }
                        arrivals.append(person)
                        
        return sorted(arrivals, key=lambda x: x['arrival_time'])
