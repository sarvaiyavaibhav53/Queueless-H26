from sqlalchemy import Column, Integer, String, Float
from app.models.base import Base

class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, index=True, nullable=False) # e.g. LL, DL_RENEWAL
    name = Column(String(100), nullable=False)
    mean_minutes = Column(Float, nullable=False)
    std_minutes = Column(Float, nullable=True)
