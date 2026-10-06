import json
from typing import Dict, List, Optional
from fastapi import WebSocket
from app.models.token import QueueToken, TokenStatus
from sqlalchemy.orm import Session

class QueueConnectionManager:
    def __init__(self):
        # Maps "officeId_serviceId" to list of WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, office_id: int, service_id: int):
        await websocket.accept()
        key = f"{office_id}_{service_id}"
        if key not in self.active_connections:
            self.active_connections[key] = []
        self.active_connections[key].append(websocket)

    def disconnect(self, websocket: WebSocket, office_id: int, service_id: int):
        key = f"{office_id}_{service_id}"
        if key in self.active_connections:
            if websocket in self.active_connections[key]:
                self.active_connections[key].remove(websocket)
            if not self.active_connections[key]:
                del self.active_connections[key]

    async def broadcast_queue_update(self, event: str, office_id: int, service_id: int, updated_token_id: Optional[int] = None):
        key = f"{office_id}_{service_id}"
        if key not in self.active_connections or not self.active_connections[key]:
            return # No one is listening

        from app.db.database import SessionLocal
        db = SessionLocal()
        try:
            # Query all waiting tokens to recalculate positions safely in memory
            waiting_tokens = db.query(QueueToken).filter(
                QueueToken.office_id == office_id,
                QueueToken.service_id == service_id,
                QueueToken.status == TokenStatus.WAITING
            ).order_by(QueueToken.created_at.asc()).all()

            token_data = []
            for i, t in enumerate(waiting_tokens):
                token_data.append({
                    "token_id": t.id,
                    "token_number": t.token_number,
                    "status": t.status.value,
                    "position": i + 1,
                    "people_ahead": i,
                    "predicted_wait_minutes": t.predicted_wait_minutes
                })

            # Also include the token that triggered this event if it is no longer WAITING
            if updated_token_id:
                updated_token = db.query(QueueToken).filter(QueueToken.id == updated_token_id).first()
                if updated_token and updated_token.status != TokenStatus.WAITING:
                    token_data.append({
                        "token_id": updated_token.id,
                        "token_number": updated_token.token_number,
                        "status": updated_token.status.value,
                        "position": 0,
                        "people_ahead": 0,
                        "predicted_wait_minutes": updated_token.predicted_wait_minutes
                    })

            message = {
                "event": event,
                "office_id": office_id,
                "service_id": service_id,
                "queue": {
                    "waiting_count": len(waiting_tokens)
                },
                "tokens": token_data
            }
        finally:
            db.close()
        
        msg_json = json.dumps(message)
        dead_connections = []
        
        for connection in self.active_connections[key]:
            try:
                await connection.send_text(msg_json)
            except Exception:
                dead_connections.append(connection)
                
        for dc in dead_connections:
            self.disconnect(dc, office_id, service_id)

ws_manager = QueueConnectionManager()
