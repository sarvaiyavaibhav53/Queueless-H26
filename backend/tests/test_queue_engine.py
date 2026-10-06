import pytest
from sqlalchemy.orm import Session
from datetime import date, datetime
from app.db.database import SessionLocal, engine
from app.models.base import Base
from app.models.office import Office
from app.models.service import Service
from app.models.token import TokenStatus, QueueToken
from app.models.queue_counter import QueueCounter
from app.services.queue_service import queue_service, QueueServiceError

@pytest.fixture(scope="module")
def db_session():
    # We will use the same test DB for simplicity, 
    # but we'll clean up the relevant tables for isolation.
    db = SessionLocal()
    
    # Cleanup before tests
    db.query(QueueToken).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    
    yield db
    
    # Cleanup after tests
    db.query(QueueToken).delete()
    db.query(QueueCounter).delete()
    db.query(Service).delete()
    db.query(Office).delete()
    db.commit()
    db.close()

@pytest.fixture(scope="module")
def setup_data(db_session: Session):
    office = Office(code="TEST", name="Test Office", counters=3, demand_multiplier=1.0)
    service = Service(code="SRV", name="Test Service", mean_minutes=10.0)
    db_session.add(office)
    db_session.add(service)
    db_session.commit()
    db_session.refresh(office)
    db_session.refresh(service)
    return {"office": office, "service": service}

def test_token_creation(db_session: Session, setup_data):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # Test token creation
    token1 = queue_service.generate_token(db_session, office_id, service_id)
    assert token1 is not None
    assert token1.token_number == "TEST-SRV-001"
    assert token1.status == TokenStatus.WAITING
    assert token1.created_at is not None
    
    # Verify uniqueness and increment
    token2 = queue_service.generate_token(db_session, office_id, service_id)
    assert token2.token_number == "TEST-SRV-002"

def test_queue_position(db_session: Session, setup_data):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # We currently have 2 tokens waiting (from previous test)
    # Get all tokens to identify them
    tokens = db_session.query(QueueToken).order_by(QueueToken.id).all()
    token1 = tokens[-2]
    token2 = tokens[-1]
    
    pos1 = queue_service.get_queue_position(db_session, token1.id)
    pos2 = queue_service.get_queue_position(db_session, token2.id)
    
    assert pos1 == 1
    assert pos2 == 2
    
    # If token1 is called, token2 position should become 1
    queue_service.call_next_token(db_session, office_id, service_id)
    
    db_session.refresh(token1)
    db_session.refresh(token2)
    assert token1.status == TokenStatus.CALLED
    
    pos2_new = queue_service.get_queue_position(db_session, token2.id)
    assert pos2_new == 1

def test_state_transitions(db_session: Session, setup_data):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # Token 1 is in CALLED state.
    tokens = db_session.query(QueueToken).order_by(QueueToken.id).all()
    token1 = tokens[-2]
    
    # CALLED -> SERVING
    token_serving = queue_service.start_serving(db_session, token1.id)
    assert token_serving.status == TokenStatus.SERVING
    assert token_serving.serving_at is not None
    
    # SERVING -> COMPLETED
    token_completed = queue_service.complete_token(db_session, token1.id)
    assert token_completed.status == TokenStatus.COMPLETED
    assert token_completed.completed_at is not None

def test_invalid_transitions(db_session: Session, setup_data):
    tokens = db_session.query(QueueToken).order_by(QueueToken.id).all()
    token1 = tokens[-2] # Currently COMPLETED
    
    # COMPLETED -> WAITING (should fail)
    with pytest.raises(QueueServiceError):
        queue_service._transition_state(token1, TokenStatus.WAITING, datetime.now())

    # COMPLETED -> SERVING (should fail)
    with pytest.raises(QueueServiceError):
        queue_service.start_serving(db_session, token1.id)

def test_cancel_and_skip(db_session: Session, setup_data):
    office_id = setup_data["office"].id
    service_id = setup_data["service"].id
    
    # Token 2 is WAITING. Cancel it.
    tokens = db_session.query(QueueToken).order_by(QueueToken.id).all()
    token2 = tokens[-1]
    
    token_cancelled = queue_service.cancel_token(db_session, token2.id)
    assert token_cancelled.status == TokenStatus.CANCELLED
    
    # Create Token 3
    token3 = queue_service.generate_token(db_session, office_id, service_id)
    assert token3.token_number == "TEST-SRV-003"
    
    # Call Token 3
    called_token = queue_service.call_next_token(db_session, office_id, service_id)
    assert called_token.id == token3.id
    
    # Skip Token 3
    skipped_token = queue_service.skip_token(db_session, token3.id)
    assert skipped_token.status == TokenStatus.SKIPPED
