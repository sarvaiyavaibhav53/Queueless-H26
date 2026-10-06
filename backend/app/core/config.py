import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "QueueLess Government Office"
    # Fallback to local default if env file doesn't override
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/queueless-gevermnent-office")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-key-for-development")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 24 hours

    class Config:
        env_file = ".env"

settings = Settings()
