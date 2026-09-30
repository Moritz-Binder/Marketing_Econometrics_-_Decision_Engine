import pytest
from unittest.mock import MagicMock
from src.agents.nodes.auditor import CausalAuditorNode

def test_causal_auditor_node_ab_audit():
    mock_auditor = MagicMock()
    mock_calibrator = MagicMock()
    
    mock_auditor.audit_sample_ratio_mismatch.return_value = {"srm_detected": True, "p_value": 0.001}
    
    node = CausalAuditorNode(mock_auditor, mock_calibrator)
    
    state = {
        "user_intent": "AUDIT_EXPERIMENT",
        "extracted_parameters": {
            "experiment_type": "A/B",
            "control_traffic": 10000,
            "variant_traffic": 9000,
            "expected_control_ratio": 0.5
        }
    }
    
    update = node(state)
    
    assert "audit_results" in update
    assert update["audit_results"]["srm_audit"]["srm_detected"] is True
    mock_auditor.audit_sample_ratio_mismatch.assert_called_once_with(
        observed_control=10000,
        observed_variant=9000,
        expected_control_ratio=0.5
    )

def test_causal_auditor_node_geo_audit():
    mock_auditor = MagicMock()
    mock_calibrator = MagicMock()
    mock_geo_auditor = MagicMock()
    
    mock_geo_auditor.compute_empirical_mde.return_value = {"empirical_mde_relative": 0.12}
    
    node = CausalAuditorNode(mock_auditor, mock_calibrator, geo_auditor=mock_geo_auditor)
    
    state = {
        "user_intent": "AUDIT_EXPERIMENT",
        "extracted_parameters": {
            "experiment_type": "Geo",
            "historical_control": [100, 105, 95],
            "historical_treatment": [102, 107, 98],
            "test_duration_days": 14
        }
    }
    
    update = node(state)
    
    assert "audit_results" in update
    assert "geo_audit" in update["audit_results"]
    assert update["audit_results"]["geo_audit"]["empirical_mde_relative"] == 0.12
    mock_geo_auditor.simulate_null_drift.assert_called_once()
    mock_geo_auditor.compute_empirical_mde.assert_called_once()

def test_causal_auditor_node_calibration():
    mock_auditor = MagicMock()
    mock_calibrator = MagicMock()
    
    mock_calibrator.calibrate_channel_prior.return_value = {
        "posterior_mean": 1.45,
        "posterior_sd": 0.15
    }
    
    node = CausalAuditorNode(mock_auditor, mock_calibrator)
    
    state = {
        "user_intent": "CALIBRATION",
        "target_channels": ["TV"],
        "extracted_parameters": {
            "prior_mean": 2.0,
            "prior_sd": 0.5,
            "lift_estimate": 1.2,
            "lift_se": 0.1,
            "alpha": 0.5,
            "test_duration_weeks": 4
        }
    }
    
    # Expected tail mult = (1 / (1 - 0.5)) / ( (1 - 0.5^4) / (1 - 0.5) )
    # total = 2.0, captured = 1.875, mult = 2.0 / 1.875 = 1.0666
    # corrected lift = 1.2 * 1.0666 = 1.28
    update = node(state)
    
    assert "calibration_results" in update
    assert update["calibration_results"]["channel"] == "TV"
    assert update["calibration_results"]["posterior_mean"] == 1.45
    
    # Check if correction was applied
    mock_calibrator.calibrate_channel_prior.assert_called_once()
    called_kwargs = mock_calibrator.calibrate_channel_prior.call_args.kwargs
    assert called_kwargs["lift_estimate"] > 1.25 # Asserting it scaled from 1.2 to ~1.28
