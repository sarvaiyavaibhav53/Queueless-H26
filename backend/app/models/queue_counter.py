from sqlalchemy import Column, Integer, Date, ForeignKey, UniqueConstraint
from app.models.base import Base
from datetime import date

class QueueCounter(Base):
    __tablename__ = "queue_counters"

    id = Column(Integer, primary_key=True, index=True)
    office_id = Column(Integer, ForeignKey("offices.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    date = Column(Date, default=date.today, nullable=False)
    current_number = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('office_id', 'service_id', 'date', name='uq_office_service_date'),
    )
