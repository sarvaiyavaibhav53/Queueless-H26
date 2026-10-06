from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.ws_manager import ws_manager
from app.models.office import Office
from app.models.service import Service
import asyncio
from app.api.deps import get_current_user_ws

router = APIRouter(prefix="/ws/queues", tags=["websockets"])

@router.websocket("/{office_id}/{service_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    office_id: int, 
    service_id: int, 
    db: Session = Depends(get_db),
    user = Depends(get_current_user_ws)
):
    # Authenticate WebSocket
    if not user:
        await websocket.close(code=1008, reason="Authentication failed")
        return
        
    # Validate office and service
    office = db.query(Office).filter(Office.id == office_id).first()
    service = db.query(Service).filter(Service.id == service_id).first()
    
    if not office or not service:
        await websocket.close(code=1008, reason="Invalid office or service ID")
        return
        
    await ws_manager.connect(websocket, office_id, service_id)
    try:
        while True:
            # We just wait for incoming messages if any, though clients might just listen.
            # Usually we need a simple receive loop to keep the connection alive and detect disconnects.
            data = await websocket.receive_text()
            # If client sends a ping or message, we can ignore it or echo it.
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, office_id, service_id)
    except Exception:
        ws_manager.disconnect(websocket, office_id, service_id)
