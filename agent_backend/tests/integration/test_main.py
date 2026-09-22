import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch

@pytest.fixture
def mock_lifespan_deps():
    with patch("src.api.main.setup_telemetry") as mock_telemetry, \
         patch("src.api.main.get_data_engine") as mock_data_engine, \
         patch("src.api.main.get_mede_workflow") as mock_workflow:
        yield {
            "telemetry": mock_telemetry,
            "data_engine": mock_data_engine,
            "workflow": mock_workflow
        }

@pytest.fixture
def client(mock_lifespan_deps):
    with TestClient(app) as test_client:
        yield test_client

def test_lifespan_startup_warmup(client, mock_lifespan_deps):
    mock_lifespan_deps["telemetry"].assert_called_once()
    mock_lifespan_deps["data_engine"].assert_called_once()
    mock_lifespan_deps["workflow"].assert_called_once()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_prometheus_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "python_info" in response.text or "process_cpu_seconds_total" in response.text

def test_cors_headers(client):
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
