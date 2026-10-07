from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api import predictions, queues, tokens, ws, offices, services, analytics, auth, interventions
from app.services.prediction_service import prediction_service
from app.db.database import get_db

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

app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(queues.router)
app.include_router(tokens.router)
app.include_router(ws.router)
app.include_router(offices.router)
app.include_router(services.router)
app.include_router(analytics.router)
app.include_router(interventions.router)

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
