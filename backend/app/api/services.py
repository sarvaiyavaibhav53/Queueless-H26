from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.schemas.service import ServiceResponse
from app.models.service import Service

router = APIRouter(prefix="/api/v1/services", tags=["services"])

@router.get("", response_model=List[ServiceResponse])
def get_services(db: Session = Depends(get_db)):
    services = db.query(Service).all()
    return services
