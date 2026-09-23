import pytest
from pydantic import ValidationError
from src.api.v1.schemas import MetricCorrection, ExperimentAuditRequest

def test_metric_correction_risk_level_validation():
    # Valid risk level
    correction = MetricCorrection(
        metric="ROAS",
        user_assumption="Linear scaling",
        mathematical_reality="Diminishing returns",
        risk_level="High",
        capital_at_risk_eur=100000.0
    )
    assert correction.risk_level == "High"
    
    # Invalid risk level should raise ValidationError
    with pytest.raises(ValidationError):
        MetricCorrection(
            metric="ROAS",
            user_assumption="Linear scaling",
            mathematical_reality="Diminishing returns",
            risk_level="Severe", # Not in Literal
            capital_at_risk_eur=100000.0
        )

def test_experiment_audit_request_defaults():
    # Test that default values are populated correctly
    request = ExperimentAuditRequest(
        control_traffic=1000,
        variant_traffic=1000,
        control_conversions=50,
        variant_conversions=75
    )
    assert request.planned_control_ratio == 0.5
    assert request.alpha == 0.05
