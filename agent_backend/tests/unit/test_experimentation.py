import numpy as np
import pytest
from src.tools.experimentation import ExperimentAuditor

def test_audit_srm_no_mismatch():
    # 50/50 split, perfectly balanced
    result = ExperimentAuditor.audit_sample_ratio_mismatch(
        observed_control=5000, 
        observed_variant=5050, 
        expected_control_ratio=0.5
    )
    assert result["srm_detected"] is False
    assert result["severity"] == "None"
    assert result["p_value"] > 0.05

def test_audit_srm_with_mismatch():
    # 50/50 expected, but clearly skewed (4k vs 6k)
    result = ExperimentAuditor.audit_sample_ratio_mismatch(
        observed_control=4000, 
        observed_variant=6000, 
        expected_control_ratio=0.5,
        alpha=0.01
    )
    assert result["srm_detected"] is True
    assert result["severity"] == "Critical"
    assert result["p_value"] < 0.01

def test_statistical_power():
    # Baseline 10% conversion, 20% relative lift (-> 12% conversion)
    # With 10,000 samples per arm, we should be well powered (>80%)
    res_powered = ExperimentAuditor.calculate_statistical_power(
        sample_size_per_arm=10000,
        baseline_p=0.10,
        relative_lift=0.20
    )
    assert res_powered["is_underpowered"] is False
    assert res_powered["power"] > 0.80
    
    # With only 500 samples per arm, we should be underpowered for a 20% lift
    res_under = ExperimentAuditor.calculate_statistical_power(
        sample_size_per_arm=500,
        baseline_p=0.10,
        relative_lift=0.20
    )
    assert res_under["is_underpowered"] is True
    assert res_under["power"] < 0.80

def test_calculate_required_sample_size():
    # To detect a 10% relative lift on a 5% baseline (so 5.5% conversion)
    n = ExperimentAuditor.calculate_required_sample_size(
        baseline_p=0.05,
        min_relative_lift=0.10,
        target_power=0.80
    )
    # Sample size should be roughly 30k per arm
    assert n > 25000 and n < 35000

def test_geo_experiment_auditor():
    from src.tools.experimentation import GeoExperimentAuditor
    
    # Create two highly correlated series (Control and Treatment pre-period)
    t = np.arange(100)
    base = 1000 + 50 * np.sin(t / 5)
    hist_c = base + np.random.normal(0, 10, 100)
    # Treatment is structurally 1.5x of Control
    hist_t = base * 1.5 + np.random.normal(0, 10, 100)  
    
    drift_dist = GeoExperimentAuditor.simulate_null_drift(
        historical_control=hist_c, 
        historical_treatment=hist_t, 
        test_duration_days=14, 
        n_bootstraps=500, 
        random_seed=42
    )
    
    assert len(drift_dist) == 500
    
    # Mean empirical drift should be near 0 if variance is effectively reduced
    assert abs(np.mean(drift_dist)) < 0.05
    
    mde_result = GeoExperimentAuditor.compute_empirical_mde(drift_dist)
    
    assert "mde_relative" in mde_result
    assert mde_result["mde_relative"] > 0
    assert mde_result["two_tailed_ci_95"][0] < 0
    assert mde_result["two_tailed_ci_95"][1] > 0
