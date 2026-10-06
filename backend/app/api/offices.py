from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.schemas.office import OfficeResponse
from app.models.office import Office

router = APIRouter(prefix="/api/v1/offices", tags=["offices"])

@router.get("", response_model=List[OfficeResponse])
def get_offices(db: Session = Depends(get_db)):
    offices = db.query(Office).all()
    return offices
