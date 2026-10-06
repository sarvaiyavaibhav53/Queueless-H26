class CounterManager:
    def __init__(self, config):
        self.counters_free = {o: config['offices'][o]['counters'] for o in config['offices']}
        self.config = config
        
    def is_free(self, office_id):
        return self.counters_free[office_id] > 0
        
    def get_total(self, office_id):
        return self.config['offices'][office_id]['counters']
        
    def allocate(self, office_id):
        if self.is_free(office_id):
            self.counters_free[office_id] -= 1
            return True
        return False
        
    def release(self, office_id):
        self.counters_free[office_id] += 1
