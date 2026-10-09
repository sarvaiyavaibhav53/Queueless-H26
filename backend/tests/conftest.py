import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
import pytest
from app.main import app
from app.api.deps import require_citizen, require_employee, require_admin, get_current_user, get_current_user_ws
from app.models.user import User, UserRole
from app.db.database import SessionLocal

@pytest.fixture(autouse=True)
def override_auth_dependencies():
    # Provide a mock admin user for all existing tests to pass
    db = SessionLocal()
    user = db.query(User).filter(User.email == "test@example.com").first()
    if not user:
        user = User(email="test@example.com", password_hash="dummy", role=UserRole.admin, is_active=True)
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()
    
    def mock_user():
        return user

    from fastapi import WebSocket, Depends
    from sqlalchemy.orm import Session
    from app.db.database import get_db
    async def mock_ws_user(websocket: WebSocket, db: Session = Depends(get_db)):
        return user
        
    app.dependency_overrides[require_citizen] = mock_user
    app.dependency_overrides[require_employee] = mock_user
    app.dependency_overrides[require_admin] = mock_user
    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[get_current_user_ws] = mock_ws_user
    
    yield
    
    app.dependency_overrides.clear()
