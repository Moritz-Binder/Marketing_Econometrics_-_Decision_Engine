from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Marketing Econometrics & Decision Engine (MEDE)"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # AI Models
    GEMINI_API_KEY: str
    ROUTER_MODEL: str = "gemini-1.5-flash"
    REASONING_MODEL: str = "gemini-1.5-pro"
    
    # Security
    API_KEY_SECRET: str = "mede-local-dev-key"

    # Data
    GOLD_DATA_PATH: str = "/app/data/gold/marketing_mix_gold.parquet"
    
    # Telemetry/Tracking
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    PROMETHEUS_PORT: int = 9090

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of the Settings object.
    """
    return Settings()