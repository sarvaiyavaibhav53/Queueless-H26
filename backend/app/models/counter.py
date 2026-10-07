import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.models.base import Base


class CounterStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    UNAVAILABLE = "UNAVAILABLE"


class Counter(Base):
    """
    Physical service desk in an office.
    Strictly follows locked Step 2A physical counter architecture:
    - id: Primary Key
    - office_id: Foreign key to offices.id
    - name: Human-readable counter desk name (e.g. 'Counter 1')
    - status: ACTIVE | UNAVAILABLE (Serving is NOT a counter status)
    - assigned_service_id: Foreign key to services.id (must be present in counter_services)
    - status_changed_at: Timezone-aware UTC timestamp
    """
    __tablename__ = "counters"

    id = Column(Integer, primary_key=True, index=True)
    office_id = Column(Integer, ForeignKey("offices.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    status = Column(SQLEnum(CounterStatus), default=CounterStatus.ACTIVE, nullable=False)
    assigned_service_id = Column(Integer, ForeignKey("services.id"), nullable=False, index=True)
    status_changed_at = Column(DateTime(timezone=True), nullable=False)

    office = relationship("Office")
    assigned_service = relationship("Service", foreign_keys=[assigned_service_id])
    counter_services = relationship("CounterService", back_populates="counter", cascade="all, delete-orphan")
    tokens = relationship("QueueToken", back_populates="counter")


class CounterService(Base):
    """
    Represents service eligibility for a physical counter desk.
    Composite primary key prevents duplicate eligibility mappings.
    """
    __tablename__ = "counter_services"

    counter_id = Column(Integer, ForeignKey("counters.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), primary_key=True, nullable=False)

    counter = relationship("Counter", back_populates="counter_services")
    service = relationship("Service")
