from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.models.base import Base

class TokenStatus(str, enum.Enum):
    WAITING = "WAITING"
    CALLED = "CALLED"
    SERVING = "SERVING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

class QueueToken(Base):
    __tablename__ = "queue_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_number = Column(String(20), nullable=False, index=True)
    
    office_id = Column(Integer, ForeignKey("offices.id"), nullable=False)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    
    status = Column(SQLEnum(TokenStatus), default=TokenStatus.WAITING, nullable=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    called_at = Column(DateTime(timezone=True), nullable=True)
    serving_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    skipped_at = Column(DateTime(timezone=True), nullable=True)
    
    # Optional prediction field if we want to store it (as per prompt instructions)
    predicted_wait_minutes = Column(Integer, nullable=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    office = relationship("Office")
    service = relationship("Service")
    user = relationship("User", back_populates="tokens")
