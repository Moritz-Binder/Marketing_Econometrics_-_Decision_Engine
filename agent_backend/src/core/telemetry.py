import os
from prometheus_client import generate_latest, REGISTRY
from src.core.config import get_settings

def setup_telemetry() -> None:
    """
    Initializes Langfuse instrumentation for LangChain/LangGraph 
    and prepares Prometheus metrics.
    """
    settings = get_settings()

    # Only configure Langfuse if the keys are actually provided
    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

def get_prometheus_metrics() -> bytes:
    """
    Generates the Prometheus text scrapable output.
    """
    return generate_latest(REGISTRY)