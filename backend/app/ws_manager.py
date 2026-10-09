"""
PRAVAAH — Real-Time WebSocket Manager (Invalidation-Only Contract).

Step 2E.5A Guarantees:
- Invalidation hints ONLY (no token lists, no citizen PII, no ETA predictions).
- Event contract:
    {
        "type": "queue.updated" | "counter.status_changed",
        "office_id": <int>,
        "service_id": <int>,
        "occurred_at": "<ISO-8601 UTC>"
    }
- Triggered post-commit.
- Clients treat events as invalidation triggers to refetch REST endpoints.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger("pravaah.ws_manager")


class QueueConnectionManager:
    def __init__(self):
        # Maps "officeId_serviceId" -> list of active WebSockets
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

    async def broadcast_invalidation(
        self,
        event_type: str,
        office_id: int,
        service_id: int,
        occurred_at: Optional[str] = None,
    ):
        """
        Broadcasts a minimal, secure invalidation hint to all connected clients
        subscribed to (office_id, service_id).
        """
        key = f"{office_id}_{service_id}"
        if key not in self.active_connections or not self.active_connections[key]:
            return

        now_iso = occurred_at or datetime.now(timezone.utc).isoformat()
        payload = {
            "type": event_type,
            "office_id": office_id,
            "service_id": service_id,
            "occurred_at": now_iso,
        }
        msg_json = json.dumps(payload)

        dead_connections = []
        for connection in list(self.active_connections[key]):
            try:
                await connection.send_text(msg_json)
            except Exception:
                dead_connections.append(connection)

        for dc in dead_connections:
            self.disconnect(dc, office_id, service_id)

    async def broadcast_queue_update(
        self,
        event: str,
        office_id: int,
        service_id: int,
        updated_token_id: Optional[int] = None,
    ):
        """
        Compatibility wrapper for queue mutations.
        Emits invalidation event 'queue.updated'.
        """
        await self.broadcast_invalidation(
            event_type="queue.updated",
            office_id=office_id,
            service_id=service_id,
        )

    async def broadcast_counter_status_changed(
        self,
        office_id: int,
        service_id: int,
        occurred_at: Optional[str] = None,
    ):
        """
        Emits invalidation event 'counter.status_changed'.
        """
        await self.broadcast_invalidation(
            event_type="counter.status_changed",
            office_id=office_id,
            service_id=service_id,
            occurred_at=occurred_at,
        )


ws_manager = QueueConnectionManager()
