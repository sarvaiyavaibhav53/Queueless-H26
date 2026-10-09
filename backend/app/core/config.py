import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "QueueLess Government Office"
    # Fallback to local sqlite if env file doesn't override
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-key-for-development")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 24 hours
    CONGESTION_QUEUE_THRESHOLD: int = int(os.getenv("CONGESTION_QUEUE_THRESHOLD", "15"))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "")

    model_config = {"env_file": ".env", "extra": "ignore"}

    def validate_security(self) -> None:
        if self.ENVIRONMENT in ["production", "prod"]:
            insecure_defaults = [
                "super-secret-key-for-development",
                "replace_with_secure_secret",
                "secret",
                "",
            ]
            if not self.JWT_SECRET or self.JWT_SECRET in insecure_defaults:
                raise ValueError(
                    "Insecure JWT_SECRET configured in production environment! "
                    "Set a cryptographically strong JWT_SECRET in production."
                )

settings = Settings()
settings.validate_security()
