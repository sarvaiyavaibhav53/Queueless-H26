"""
PRAVAAH — Simulation Events & Deterministic Event Queue.

Defines all discrete event types and priority queue ordering.
Guarantees:
- Strict deterministic ordering: primary by simulation time, secondary by EventPriority,
  tertiary by monotonic sequence number.
- No second-by-second polling loops; pure event-driven heap execution.
"""

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any


@unique
class EventType(str, Enum):
    CUSTOMER_ARRIVAL = "CUSTOMER_ARRIVAL"
    SERVICE_START = "SERVICE_START"
    SERVICE_COMPLETION = "SERVICE_COMPLETION"
    COUNTER_UNAVAILABLE = "COUNTER_UNAVAILABLE"
    COUNTER_AVAILABLE = "COUNTER_AVAILABLE"
    INTERVENTION_REASSIGN = "INTERVENTION_REASSIGN"
    INTERVENTION_REQUEST = "INTERVENTION_REQUEST"
    INTERVENTION_ACTIVATED = "INTERVENTION_ACTIVATED"
    INTERVENTION_EXPIRED = "INTERVENTION_EXPIRED"
    INTERVENTION_RESTORED = "INTERVENTION_RESTORED"


# Documented priority ordering for events occurring at the exact same simulation timestamp
EVENT_PRIORITY = {
    EventType.INTERVENTION_REASSIGN: 10,
    EventType.INTERVENTION_REQUEST: 10,
    EventType.INTERVENTION_ACTIVATED: 11,
    EventType.INTERVENTION_EXPIRED: 12,
    EventType.INTERVENTION_RESTORED: 13,
    EventType.COUNTER_UNAVAILABLE: 20,
    EventType.COUNTER_AVAILABLE: 30,
    EventType.SERVICE_COMPLETION: 40,
    EventType.CUSTOMER_ARRIVAL: 50,
    EventType.SERVICE_START: 60,
}


@dataclass(order=True)
class SimEvent:
    """
    Individual discrete event in the simulation priority queue.
    Comparison tuple: (time, priority, sequence_num)
    """
    time: float
    priority: int
    sequence_num: int
    event_type: EventType = field(compare=False)
    data: Any = field(compare=False, default=None)
