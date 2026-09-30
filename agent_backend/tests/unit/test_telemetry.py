import os
from src.core.telemetry import setup_telemetry, get_prometheus_metrics

def test_setup_telemetry(monkeypatch):
    """
    Verify that telemetry correctly passes settings to os.environ.
    """
    # Force the Settings singleton to read these mocked env vars
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-sk")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    
    # We must clear the cache so `get_settings()` parses the monkeypatched env vars
    from src.core.config import get_settings
    get_settings.cache_clear()
    
    setup_telemetry()
    
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "test-pk"
    assert os.environ.get("LANGFUSE_SECRET_KEY") == "test-sk"

def test_get_prometheus_metrics():
    """
    Verify that we can scrape prometheus metrics.
    """
    metrics = get_prometheus_metrics()
    assert isinstance(metrics, bytes)
    # The default registry will have some base metrics generated, so it shouldn't be empty
    assert len(metrics) > 0