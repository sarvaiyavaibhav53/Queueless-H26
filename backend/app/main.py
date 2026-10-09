import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.api import predictions, queues, tokens, ws, offices, services, analytics, auth, interventions, admin, counters
from app.services.prediction_service import prediction_service
from app.db.database import get_db

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def get_cors_origins() -> list[str]:
    """
    Resolves CORS origins:
    - Default localhost and 127.0.0.1 development ports (3000, 5173, 8080, 4173).
    - Merges with CORS_ORIGINS environment variable (comma-separated).
    - Excludes opaque 'null' and wildcard '*' (not safe with allow_credentials=True).
    """
    origins = list(DEFAULT_CORS_ORIGINS)
    custom_origins_env = settings.CORS_ORIGINS or os.getenv("CORS_ORIGINS", "")
    if custom_origins_env:
        for origin in custom_origins_env.split(","):
            cleaned = origin.strip()
            if cleaned and cleaned.lower() != "null" and cleaned != "*":
                if cleaned not in origins:
                    origins.append(cleaned)
    return origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML model on startup
    try:
        prediction_service.load_model()
    except Exception as e:
        print(f"Warning: Model failed to load on startup: {e}")
    yield
    # Cleanup on shutdown if needed
    pass

app = FastAPI(title="QueueLess API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(queues.router)
app.include_router(tokens.router)
app.include_router(ws.router)
app.include_router(offices.router)
app.include_router(services.router)
app.include_router(analytics.router)
app.include_router(interventions.router)
app.include_router(admin.router)
app.include_router(counters.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to QueueLess API"}

@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
