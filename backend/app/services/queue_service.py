from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timezone
from app.models.office import Office
from app.models.service import Service
from app.models.token import QueueToken, TokenStatus
import threading
from app.models.counter import Counter, CounterService, CounterStatus
from app.models.queue_counter import QueueCounter

_queue_lock = threading.Lock()

class QueueServiceError(Exception):
    pass

class CounterNotFoundError(QueueServiceError):
    pass

class CounterConflictError(QueueServiceError):
    pass

class CounterValidationError(QueueServiceError):
    pass

class QueueService:
    def _get_utc_now(self):
        return datetime.now(timezone.utc)

    def generate_token(self, db: Session, office_id: int, service_id: int) -> QueueToken:
        """
        Generates a unique queue token deterministically using row-level locking (FOR UPDATE).
        """
        with _queue_lock:
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

    def call_next_token(
        self,
        db: Session,
        office_id: int,
        service_id: int,
        counter_id: int = None,
    ) -> QueueToken:
        """
        Gets the next waiting token and changes its status to CALLED.
        Validates all 9 invariants when counter_id is provided.
        """
        with _queue_lock:
            assigned_counter_id = None

            if counter_id is not None:
                # 1. Counter exists
                counter = db.query(Counter).filter(Counter.id == counter_id).with_for_update().first()
                if not counter:
                    raise CounterNotFoundError(f"Counter with ID {counter_id} not found.")

                # 2. Belongs to requested office
                if counter.office_id != office_id:
                    raise CounterValidationError(
                        f"Counter {counter_id} belongs to office {counter.office_id}, not {office_id}."
                    )

                # 3. Status is ACTIVE
                if counter.status != CounterStatus.ACTIVE:
                    raise CounterValidationError(
                        f"Counter '{counter.name}' is UNAVAILABLE and cannot call tokens."
                    )

                # 4. Assigned to requested service
                if counter.assigned_service_id != service_id:
                    raise CounterValidationError(
                        f"Counter '{counter.name}' is assigned to service {counter.assigned_service_id}, not {service_id}."
                    )

                # 5. Eligible to handle service
                is_eligible = (
                    db.query(CounterService)
                    .filter(
                        CounterService.counter_id == counter.id,
                        CounterService.service_id == service_id,
                    )
                    .first()
                )
                if not is_eligible:
                    raise CounterValidationError(
                        f"Counter '{counter.name}' is not eligible to serve service {service_id}."
                    )

                # 6. Counter is idle (no active token in CALLED or SERVING)
                active_token = (
                    db.query(QueueToken)
                    .filter(
                        QueueToken.counter_id == counter.id,
                        QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
                    )
                    .first()
                )
                if active_token:
                    raise CounterConflictError(
                        f"Counter '{counter.name}' is already busy with active token '{active_token.token_number}'."
                    )

                assigned_counter_id = counter.id
            else:
                # Fallback for unassigned caller: pick first idle active counter assigned to this service
                active_counters = (
                    db.query(Counter)
                    .filter(
                        Counter.office_id == office_id,
                        Counter.assigned_service_id == service_id,
                        Counter.status == CounterStatus.ACTIVE,
                    )
                    .all()
                )
                for c in active_counters:
                    busy = (
                        db.query(QueueToken)
                        .filter(
                            QueueToken.counter_id == c.id,
                            QueueToken.status.in_([TokenStatus.CALLED, TokenStatus.SERVING]),
                        )
                        .first()
                    )
                    if not busy:
                        assigned_counter_id = c.id
                        break

            # 7. Find oldest WAITING token using locked FIFO order (created_at ASC, id ASC)
            token = (
                db.query(QueueToken)
                .filter(
                    QueueToken.office_id == office_id,
                    QueueToken.service_id == service_id,
                    QueueToken.status == TokenStatus.WAITING,
                )
                .order_by(QueueToken.created_at.asc(), QueueToken.id.asc())
                .with_for_update()
                .first()
            )

            if not token:
                return None

            token.status = TokenStatus.CALLED
            token.counter_id = assigned_counter_id
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
