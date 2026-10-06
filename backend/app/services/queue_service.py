from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timezone
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
from app.models.queue_counter import QueueCounter

class QueueServiceError(Exception):
    pass

class QueueService:
    def _get_utc_now(self):
        return datetime.now(timezone.utc)

    def generate_token(self, db: Session, office_id: int, service_id: int) -> QueueToken:
        """
        Generates a unique queue token deterministically using row-level locking (FOR UPDATE).
        """
        today = date.today()
        
        # Verify office and service exist
        office = db.query(Office).filter(Office.id == office_id).first()
        service = db.query(Service).filter(Service.id == service_id).first()
        
        if not office or not service:
            raise QueueServiceError("Invalid office or service")

        from sqlalchemy.exc import IntegrityError
        # 1. Lock or create the counter row for today
        counter = db.query(QueueCounter).filter(
            QueueCounter.office_id == office_id,
            QueueCounter.service_id == service_id,
            QueueCounter.date == today
        ).with_for_update().first()

        if not counter:
            try:
                # Use a nested transaction (savepoint) so we can catch the race condition
                with db.begin_nested():
                    counter = QueueCounter(
                        office_id=office_id, 
                        service_id=service_id, 
                        date=today, 
                        current_number=0
                    )
                    db.add(counter)
            except IntegrityError:
                # Another thread already created it, let's just proceed
                pass
            
            # Re-fetch with lock
            counter = db.query(QueueCounter).filter(
                QueueCounter.office_id == office_id,
                QueueCounter.service_id == service_id,
                QueueCounter.date == today
            ).with_for_update().first()

        counter.current_number += 1
        
        # Token format: [OfficeCode]-[ServiceCode]-[Number] (e.g., RJT-LL-001)
        token_number_str = f"{office.code}-{service.code}-{counter.current_number:03d}"
        
        token = QueueToken(
            token_number=token_number_str,
            office_id=office_id,
            service_id=service_id,
            status=TokenStatus.WAITING,
            created_at=self._get_utc_now()
        )
        
        db.add(token)
        db.commit()
        db.refresh(token)
        return token

    def get_queue_position(self, db: Session, token_id: int) -> int:
        """
        Calculates the queue position based on the number of waiting tokens created before this one.
        """
        token = db.query(QueueToken).filter(QueueToken.id == token_id).first()
        if not token:
            raise QueueServiceError("Token not found")
            
        if token.status != TokenStatus.WAITING:
            return 0 # Not waiting, no position
            
        # Count tokens in WAITING state created before this one for the same office/service
        position = db.query(func.count(QueueToken.id)).filter(
            QueueToken.office_id == token.office_id,
            QueueToken.service_id == token.service_id,
            QueueToken.status == TokenStatus.WAITING,
            QueueToken.created_at < token.created_at
        ).scalar()
        
        return position + 1

    def call_next_token(self, db: Session, office_id: int, service_id: int) -> QueueToken:
        """
        Gets the next waiting token and changes its status to CALLED.
        """
        # Find the oldest WAITING token
        token = db.query(QueueToken).filter(
            QueueToken.office_id == office_id,
            QueueToken.service_id == service_id,
            QueueToken.status == TokenStatus.WAITING
        ).order_by(QueueToken.created_at.asc()).with_for_update().first()
        
        if not token:
            return None
            
        token.status = TokenStatus.CALLED
        token.called_at = self._get_utc_now()
        db.commit()
        db.refresh(token)
        return token

    def _transition_state(self, token: QueueToken, new_status: TokenStatus, current_time: datetime):
        valid_transitions = {
            TokenStatus.WAITING: [TokenStatus.CALLED, TokenStatus.CANCELLED],
            TokenStatus.CALLED: [TokenStatus.SERVING, TokenStatus.SKIPPED],
            TokenStatus.SERVING: [TokenStatus.COMPLETED],
            TokenStatus.COMPLETED: [],
            TokenStatus.CANCELLED: [],
            TokenStatus.SKIPPED: []
        }

        if new_status not in valid_transitions[token.status]:
            raise QueueServiceError(f"Invalid state transition from {token.status.value} to {new_status.value}")
            
        token.status = new_status
        if new_status == TokenStatus.SERVING:
            token.serving_at = current_time
        elif new_status == TokenStatus.COMPLETED:
            token.completed_at = current_time
        elif new_status == TokenStatus.CANCELLED:
            token.cancelled_at = current_time
        elif new_status == TokenStatus.SKIPPED:
            token.skipped_at = current_time

    def start_serving(self, db: Session, token_id: int) -> QueueToken:
        token = db.query(QueueToken).filter(QueueToken.id == token_id).with_for_update().first()
        if not token:
            raise QueueServiceError("Token not found")
            
        self._transition_state(token, TokenStatus.SERVING, self._get_utc_now())
        db.commit()
        db.refresh(token)
        return token

    def complete_token(self, db: Session, token_id: int) -> QueueToken:
        token = db.query(QueueToken).filter(QueueToken.id == token_id).with_for_update().first()
        if not token:
            raise QueueServiceError("Token not found")
            
        self._transition_state(token, TokenStatus.COMPLETED, self._get_utc_now())
        db.commit()
        db.refresh(token)
        return token
        
    def cancel_token(self, db: Session, token_id: int) -> QueueToken:
        token = db.query(QueueToken).filter(QueueToken.id == token_id).with_for_update().first()
        if not token:
            raise QueueServiceError("Token not found")
            
        self._transition_state(token, TokenStatus.CANCELLED, self._get_utc_now())
        db.commit()
        db.refresh(token)
        return token

    def skip_token(self, db: Session, token_id: int) -> QueueToken:
        token = db.query(QueueToken).filter(QueueToken.id == token_id).with_for_update().first()
        if not token:
            raise QueueServiceError("Token not found")
            
        self._transition_state(token, TokenStatus.SKIPPED, self._get_utc_now())
        db.commit()
        db.refresh(token)
        return token

queue_service = QueueService()
