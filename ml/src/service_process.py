import numpy as np
import random

class ServiceProcess:
    def __init__(self, config):
        self.config = config

    def sample_duration(self, service_id):
        srv = self.config['services'][service_id]
        m, s = srv['mean_minutes'], srv['std_minutes']
        
        sigma_sq = np.log(1 + (s**2) / (m**2))
        sigma = np.sqrt(sigma_sq)
        mu = np.log(m) - sigma_sq / 2.0
        
        duration = np.random.lognormal(mu, sigma)
        
        if random.random() < self.config['long_service']['probability']:
            duration *= self.config['long_service']['multiplier']
            
        return max(0.5, duration)
