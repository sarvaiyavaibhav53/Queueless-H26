from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
from datetime import datetime
from app.db.database import get_db
from app.models.token import QueueToken, TokenStatus
from app.models.office import Office
from app.models.service import Service
from app.schemas.analytics import (
    OverallAnalyticsResponse,
    OfficeAnalyticsResponse,
    ServiceAnalyticsResponse,
    QueueAnalyticsResponse,
    MLMonitoringResponse
)
import math
from app.api.deps import require_admin
from app.models.user import User

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

def apply_date_filter(query, start_date: Optional[datetime], end_date: Optional[datetime]):
    if start_date:
        query = query.filter(QueueToken.created_at >= start_date)
    if end_date:
        query = query.filter(QueueToken.created_at <= end_date)
    return query

def _calculate_avg_wait_and_service(query):
    wait_rows = query.filter(QueueToken.serving_at.isnot(None), QueueToken.created_at.isnot(None)).with_entities(QueueToken.serving_at, QueueToken.created_at).all()
    wait_diffs = [(r[0] - r[1]).total_seconds() for r in wait_rows if r[0] and r[1]]
    avg_wait = (sum(wait_diffs) / len(wait_diffs)) / 60.0 if wait_diffs else None

    service_rows = query.filter(QueueToken.completed_at.isnot(None), QueueToken.serving_at.isnot(None)).with_entities(QueueToken.completed_at, QueueToken.serving_at).all()
    service_diffs = [(r[0] - r[1]).total_seconds() for r in service_rows if r[0] and r[1]]
    avg_service = (sum(service_diffs) / len(service_diffs)) / 60.0 if service_diffs else None

    return avg_wait, avg_service

@router.get("/overall", response_model=OverallAnalyticsResponse)
def get_overall_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
    query = db.query(QueueToken)
    query = apply_date_filter(query, start_date, end_date)
    
    total = query.count()
    if total == 0:
        return OverallAnalyticsResponse(
            total_tokens=0, waiting_tokens=0, called_tokens=0, serving_tokens=0,
            completed_tokens=0, cancelled_tokens=0, skipped_tokens=0,
            completion_rate=0.0, cancellation_rate=0.0, skip_rate=0.0
        )
        
    status_counts = query.with_entities(QueueToken.status, func.count(QueueToken.id)).group_by(QueueToken.status).all()
    counts = {s: 0 for s in TokenStatus}
    for status, count in status_counts:
        counts[status] = count
        
    avg_wait, avg_service = _calculate_avg_wait_and_service(query)

    return OverallAnalyticsResponse(
        total_tokens=total,
        waiting_tokens=counts[TokenStatus.WAITING],
        called_tokens=counts[TokenStatus.CALLED],
        serving_tokens=counts[TokenStatus.SERVING],
        completed_tokens=counts[TokenStatus.COMPLETED],
        cancelled_tokens=counts[TokenStatus.CANCELLED],
        skipped_tokens=counts[TokenStatus.SKIPPED],
        average_wait_minutes=avg_wait,
        average_service_minutes=avg_service,
        completion_rate=(counts[TokenStatus.COMPLETED] / total) * 100,
        cancellation_rate=(counts[TokenStatus.CANCELLED] / total) * 100,
        skip_rate=(counts[TokenStatus.SKIPPED] / total) * 100
    )

@router.get("/offices/{office_id}", response_model=OfficeAnalyticsResponse)
def get_office_analytics(
    office_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(status_code=404, detail="Office not found")
        
    query = db.query(QueueToken).filter(QueueToken.office_id == office_id)
    query = apply_date_filter(query, start_date, end_date)
    
    total = query.count()
    if total == 0:
        return OfficeAnalyticsResponse(
            office_id=office.id, office_code=office.code, total_tokens=0,
            completed_tokens=0, cancelled_tokens=0, skipped_tokens=0,
            current_waiting_count=0, current_serving_count=0
        )
        
    status_counts = query.with_entities(QueueToken.status, func.count(QueueToken.id)).group_by(QueueToken.status).all()
    counts = {s: 0 for s in TokenStatus}
    for status, count in status_counts:
        counts[status] = count
        
    avg_wait, avg_service = _calculate_avg_wait_and_service(query)

    return OfficeAnalyticsResponse(
        office_id=office.id,
        office_code=office.code,
        total_tokens=total,
        completed_tokens=counts[TokenStatus.COMPLETED],
        cancelled_tokens=counts[TokenStatus.CANCELLED],
        skipped_tokens=counts[TokenStatus.SKIPPED],
        current_waiting_count=counts[TokenStatus.WAITING],
        current_serving_count=counts[TokenStatus.SERVING],
        average_wait_minutes=avg_wait,
        average_service_minutes=avg_service
    )

@router.get("/services/{service_id}", response_model=ServiceAnalyticsResponse)
def get_service_analytics(
    service_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
        
    query = db.query(QueueToken).filter(QueueToken.service_id == service_id)
    query = apply_date_filter(query, start_date, end_date)
    
    total = query.count()
    if total == 0:
        return ServiceAnalyticsResponse(
            service_id=service.id, service_code=service.code, total_tokens=0,
            completed_tokens=0, cancelled_tokens=0, skipped_tokens=0,
            current_waiting_count=0
        )
        
    status_counts = query.with_entities(QueueToken.status, func.count(QueueToken.id)).group_by(QueueToken.status).all()
    counts = {s: 0 for s in TokenStatus}
    for status, count in status_counts:
        counts[status] = count
        
    avg_wait, avg_service = _calculate_avg_wait_and_service(query)

    return ServiceAnalyticsResponse(
        service_id=service.id,
        service_code=service.code,
        total_tokens=total,
        completed_tokens=counts[TokenStatus.COMPLETED],
        cancelled_tokens=counts[TokenStatus.CANCELLED],
        skipped_tokens=counts[TokenStatus.SKIPPED],
        current_waiting_count=counts[TokenStatus.WAITING],
        average_wait_minutes=avg_wait,
        average_service_minutes=avg_service
    )

@router.get("/queues/{office_id}/{service_id}", response_model=QueueAnalyticsResponse)
def get_queue_analytics(office_id: int, service_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    office = db.query(Office).filter(Office.id == office_id).first()
    if not office:
        raise HTTPException(status_code=404, detail="Office not found")
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
        
    status_counts = db.query(QueueToken.status, func.count(QueueToken.id))\
        .filter(QueueToken.office_id == office_id, QueueToken.service_id == service_id)\
        .group_by(QueueToken.status).all()
        
    counts = {s: 0 for s in TokenStatus}
    for status, count in status_counts:
        counts[status] = count
        
    return QueueAnalyticsResponse(
        waiting_count=counts[TokenStatus.WAITING],
        called_count=counts[TokenStatus.CALLED],
        serving_count=counts[TokenStatus.SERVING],
        active_counters=office.counters
    )

@router.get("/ml-monitoring", response_model=MLMonitoringResponse)
def get_ml_monitoring(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
        
    query = db.query(QueueToken)
    query = apply_date_filter(query, start_date, end_date)
    
    with_preds = query.filter(QueueToken.predicted_wait_minutes.isnot(None)).count()
    without_preds = query.filter(QueueToken.predicted_wait_minutes.is_(None)).count()
    
    if with_preds == 0:
        return MLMonitoringResponse(
            total_with_predictions=0,
            total_without_predictions=without_preds
        )
        
    avg_pred = query.filter(QueueToken.predicted_wait_minutes.isnot(None))\
        .with_entities(func.avg(QueueToken.predicted_wait_minutes)).scalar()
        
    # MAE / RMSE calculation
    # Only for tokens that have completed_at and predicted_wait_minutes
    completed = query.filter(QueueToken.completed_at.isnot(None), QueueToken.predicted_wait_minutes.isnot(None)).all()
    
    mae = None
    rmse = None
    if completed:
        errors = []
        for t in completed:
            # actual wait was until serving
            if t.serving_at and t.created_at:
                actual_wait_mins = (t.serving_at - t.created_at).total_seconds() / 60.0
                err = actual_wait_mins - t.predicted_wait_minutes
                errors.append(err)
        if errors:
            mae = sum(abs(e) for e in errors) / len(errors)
            rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
            
    return MLMonitoringResponse(
        total_with_predictions=with_preds,
        total_without_predictions=without_preds,
        average_predicted_wait_minutes=float(avg_pred) if avg_pred else None,
        mae_minutes=mae,
        rmse_minutes=rmse
    )
