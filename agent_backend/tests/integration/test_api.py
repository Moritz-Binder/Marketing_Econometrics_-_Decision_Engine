import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.v1.endpoints import router
from src.api.dependencies import get_mede_workflow

app = FastAPI()
app.include_router(router, prefix="/api/v1")

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

def test_analyze_decision_endpoint_success(client):
    mock_workflow = MagicMock()
    mock_workflow.invoke.return_value = {
        "final_brief": {
            "summary_verdict": "Test verdict.",
            "fallacies": [],
            "recommended_allocations": [],
            "methodology_caveats": []
        }
    }
    app.dependency_overrides[get_mede_workflow] = lambda: mock_workflow
    
    response = client.post("/api/v1/analyze", json={"query": "Should I scale TV?"})
    assert response.status_code == 200
    assert response.json()["summary_verdict"] == "Test verdict."

def test_analyze_decision_endpoint_missing_brief_500(client):
    mock_workflow = MagicMock()
    mock_workflow.invoke.return_value = {}  # Missing final_brief
    app.dependency_overrides[get_mede_workflow] = lambda: mock_workflow
    
    response = client.post("/api/v1/analyze", json={"query": "Should I scale TV?"})
    assert response.status_code == 500
    assert "Failed to generate decision brief" in response.json()["detail"]

def test_analyze_decision_validation_error_422(client):
    # Missing required 'query' field
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422

def test_audit_experiment_endpoint(client):
    # We don't need to mock the deterministic engine, it's fast enough
    payload = {
        "control_traffic": 10000,
        "variant_traffic": 9000, # This triggers SRM
        "control_conversions": 500,
        "variant_conversions": 480
    }
    
    response = client.post("/api/v1/audit-experiment", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["srm_detected"] is True
    assert "INVALID TEST" in data["verdict"]

@patch("src.tools.bayesian_calibrator.BayesianLiftCalibrator.calibrate_channel_prior")
def test_calibrate_channel_endpoint(mock_calibrate, client):
    mock_calibrate.return_value = {
        "posterior_mean": 1.45,
        "posterior_sd": 0.15,
        "hdi_95": [1.15, 1.75],
        "shrinkage_factor": 0.8
    }
    
    payload = {
        "channel": "TV",
        "mmm_prior_mean": 2.0,
        "mmm_prior_sd": 0.5,
        "experiment_lift_mean": 1.2,
        "experiment_lift_se": 0.1
    }
    
    response = client.post("/api/v1/calibrate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["calibrated_posterior_mean"] == 1.45
    assert data["channel"] == "TV"
