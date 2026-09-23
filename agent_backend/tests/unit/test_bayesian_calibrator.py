import pytest
import numpy as np
from src.tools.bayesian_calibrator import BayesianLiftCalibrator

def test_bayesian_calibration():
    calibrator = BayesianLiftCalibrator(random_seed=42)
    
    # Scenario: 
    # The MMM says TV ROAS is 2.5 (but the model is highly uncertain, sd=1.0)
    # The GeoExperiment says TV ROAS is 1.5 (and it's very precise, sd=0.2)
    result = calibrator.calibrate_channel_prior(
        prior_mean=2.5,
        prior_sd=1.0,
        lift_estimate=1.5,
        lift_se=0.2,
        draws=500,  # Fast sampling just for the unit test
        tune=200
    )
    
    # 1. The posterior mean should be heavily pulled towards the GeoExperiment (1.5)
    # because its standard error (0.2) is much tighter than the prior (1.0).
    assert result["posterior_mean"] < 2.0
    assert result["posterior_mean"] > 1.3
    
    # 2. The posterior uncertainty should be narrower than the prior
    assert result["posterior_sd"] < 1.0
    assert result["posterior_sd"] > 0.0
    
    # 3. Shrinkage should be highly positive since the precision drastically increased
    assert result["shrinkage_factor"] > 0.5
    
    # 4. Check HDI bounds integrity
    hdi = result["hdi_95"]
    assert len(hdi) == 2
    assert hdi[0] < hdi[1]
    assert hdi[0] < result["posterior_mean"] < hdi[1]

def test_lift_correction_engine():
    from src.tools.bayesian_calibrator import LiftCorrectionEngine
    
    # Adstock Tail Multiplier Test
    # alpha = 0.5, T = 4
    # Captured = 1 + 0.5 + 0.25 + 0.125 = 1.875
    # Total = 1 / 0.5 = 2.0
    # Multiplier = 2.0 / 1.875 = 1.0666...
    tail_mult = LiftCorrectionEngine.calculate_adstock_tail_multiplier(alpha=0.5, test_duration_weeks=4)
    assert np.isclose(tail_mult, 2.0 / 1.875)
    
    # Saturation Multiplier Test
    # V_max = 100, K = 50, n = 2
    # Baseline S=50 -> aROAS = 1.0 (50 revenue)
    # Test S=100 -> aROAS = 0.8 (80 revenue)
    # Saturation Multiplier = 1.0 / 0.8 = 1.25
    sat_mult = LiftCorrectionEngine.calculate_saturation_multiplier(
        baseline_spend=50, 
        test_spend=100, 
        v_max=100, 
        k=50, 
        n=2
    )
    assert np.isclose(sat_mult, 1.25)
    
    # Apply Corrections Test
    corrected_lift, corrected_se = LiftCorrectionEngine.apply_corrections(
        raw_lift=1.0, 
        raw_se=0.2, 
        tail_multiplier=1.5, 
        saturation_multiplier=2.0
    )
    # Total multiplier = 3.0
    assert np.isclose(corrected_lift, 3.0)
    assert np.isclose(corrected_se, 0.6)
