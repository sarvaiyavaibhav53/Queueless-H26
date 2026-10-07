"""
PRAVAAH — Simulation Results & Metrics Container.

Computes and structures all discrete-event simulation performance metrics.
Guarantees:
- Total waiting person-minutes computed strictly from queue arrival to service start.
- Clear separation between served customer waiting and unserved queue backlog.
- Immutable typed result representations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ServiceMetrics:
    """
    Per-service operational metrics over the simulation horizon.
    """
    service_id: int
    service_code: str
    customers_served: int
    waiting_person_minutes: float           # Served customers: sum(service_start - arrival)
    unserved_customers_count: int           # Count of citizens remaining in queue at horizon
    unserved_waiting_person_minutes: float  # Unserved citizens: sum(horizon - arrival)
    total_waiting_person_minutes: float     # waiting_person_minutes + unserved_waiting_person_minutes
    avg_waiting_time_minutes: float         # Average wait for served citizens (0.0 if none)
    max_waiting_time_minutes: float         # Maximum wait for served citizens (0.0 if none)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_code": self.service_code,
            "customers_served": self.customers_served,
            "waiting_person_minutes": round(self.waiting_person_minutes, 4),
            "unserved_customers_count": self.unserved_customers_count,
            "unserved_waiting_person_minutes": round(self.unserved_waiting_person_minutes, 4),
            "total_waiting_person_minutes": round(self.total_waiting_person_minutes, 4),
            "avg_waiting_time_minutes": round(self.avg_waiting_time_minutes, 4),
            "max_waiting_time_minutes": round(self.max_waiting_time_minutes, 4),
        }


@dataclass(frozen=True)
class InterventionRecord:
    """
    Simulation telemetry record for a counter intervention lifecycle.
    """
    counter_id: int
    original_service_id: int
    target_service_id: int
    requested_time_min: float
    duration_min: Optional[float]
    switching_delay_min: float
    actual_active_start_min: Optional[float]
    actual_active_end_min: Optional[float]
    actual_restored_min: Optional[float]
    actual_active_duration_min: Optional[float]
    status: str  # "RESTORED", "ACTIVE", "TERMINATED", "PENDING_REVERSION", "SWITCHING_TO_ORIGINAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counter_id": self.counter_id,
            "original_service_id": self.original_service_id,
            "target_service_id": self.target_service_id,
            "requested_time_min": round(self.requested_time_min, 4),
            "duration_min": round(self.duration_min, 4) if self.duration_min is not None else None,
            "switching_delay_min": round(self.switching_delay_min, 4),
            "actual_active_start_min": (
                round(self.actual_active_start_min, 4) if self.actual_active_start_min is not None else None
            ),
            "actual_active_end_min": (
                round(self.actual_active_end_min, 4) if self.actual_active_end_min is not None else None
            ),
            "actual_restored_min": (
                round(self.actual_restored_min, 4) if self.actual_restored_min is not None else None
            ),
            "actual_active_duration_min": (
                round(self.actual_active_duration_min, 4) if self.actual_active_duration_min is not None else None
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class SimulationResult:
    """
    Comprehensive result of a discrete-event simulation run.
    """
    horizon_minutes: float
    total_customers_served: int
    total_waiting_person_minutes: float     # Primary objective: served + unserved waiting person-minutes
    served_waiting_person_minutes: float    # Sum of waiting times of all served customers
    unserved_waiting_person_minutes: float  # Backlog accrued wait up to horizon for unserved customers
    avg_waiting_time_minutes: float         # Overall average waiting time among served customers
    max_waiting_time_minutes: float         # Maximum waiting time among served customers
    service_metrics: Dict[int, ServiceMetrics]
    queue_length_history: Tuple[Tuple[float, int, int], ...]  # (time_min, service_id, length)
    completed_citizen_ids: Tuple[int, ...]
    unserved_citizen_ids: Tuple[int, ...]
    service_waiting_times: Dict[int, Tuple[float, ...]] = field(default_factory=dict)
    intervention_records: Tuple[InterventionRecord, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "total_customers_served": self.total_customers_served,
            "total_waiting_person_minutes": round(self.total_waiting_person_minutes, 4),
            "served_waiting_person_minutes": round(self.served_waiting_person_minutes, 4),
            "unserved_waiting_person_minutes": round(self.unserved_waiting_person_minutes, 4),
            "avg_waiting_time_minutes": round(self.avg_waiting_time_minutes, 4),
            "max_waiting_time_minutes": round(self.max_waiting_time_minutes, 4),
            "services": {s_id: m.to_dict() for s_id, m in self.service_metrics.items()},
            "completed_citizens_count": len(self.completed_citizen_ids),
            "unserved_citizens_count": len(self.unserved_citizen_ids),
            "interventions": [r.to_dict() for r in self.intervention_records],
        }
