from sqlalchemy import Column, Integer, String, Float
from app.models.base import Base

class Office(Base):
    __tablename__ = "offices"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(10), unique=True, index=True, nullable=False) # e.g. RJT, AMD
    name = Column(String(100), nullable=False)
    counters = Column(Integer, default=1)
    demand_multiplier = Column(Float, default=1.0)
