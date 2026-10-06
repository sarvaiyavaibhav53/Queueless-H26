class QueueEngine:
    def __init__(self, offices):
        self.queues = {o: [] for o in offices}
    
    def enqueue(self, office_id, person):
        self.queues[office_id].append(person)
        
    def dequeue(self, office_id):
        return self.queues[office_id].pop(0) if self.queues[office_id] else None
        
    def length(self, office_id):
        return len(self.queues[office_id])
