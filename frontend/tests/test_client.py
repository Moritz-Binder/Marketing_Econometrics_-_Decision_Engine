import pytest
import httpx
from unittest.mock import patch, MagicMock
from frontend.client import MEDEClient

@pytest.fixture
def client():
    return MEDEClient(base_url="http://testserver/api/v1")

@patch("httpx.Client.get")
def test_check_health(mock_get, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    is_healthy = client.check_health()
    
    assert is_healthy is True
    mock_get.assert_called_once_with("http://testserver/health")

@patch("httpx.Client.post")
def test_analyze_decision(mock_post, client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"summary_verdict": "Increase TV spend"}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = client.analyze_decision(
        query="Should we increase TV spend?", 
        total_budget_eur=50000.0, 
        target_channels=["TV"]
    )
    
    assert result["summary_verdict"] == "Increase TV spend"
    mock_post.assert_called_once_with(
        "http://testserver/api/v1/analyze",
        json={
            "query": "Should we increase TV spend?",
            "total_budget_eur": 50000.0,
            "target_channels": ["TV"]
        }
    )

@patch("httpx.Client.post")
def test_audit_experiment(mock_post, client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"srm_detected": False}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = client.audit_experiment(1000, 1000, 50, 60)
    
    assert result["srm_detected"] is False
    mock_post.assert_called_once_with(
        "http://testserver/api/v1/audit-experiment",
        json={
            "control_traffic": 1000,
            "variant_traffic": 1000,
            "control_conversions": 50,
            "variant_conversions": 60,
            "planned_control_ratio": 0.5,
            "alpha": 0.05
        }
    )

@patch("httpx.Client.post")
def test_calibrate_channel(mock_post, client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"shrinkage_factor": 0.8}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = client.calibrate_channel("TV", 1.5, 0.2, 1.8, 0.1)
    
    assert result["shrinkage_factor"] == 0.8
    mock_post.assert_called_once_with(
        "http://testserver/api/v1/calibrate",
        json={
            "channel": "TV",
            "mmm_prior_mean": 1.5,
            "mmm_prior_sd": 0.2,
            "experiment_lift_mean": 1.8,
            "experiment_lift_se": 0.1
        }
    )
