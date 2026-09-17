import pymc as pm
import arviz as az
import numpy as np
from typing import Dict, Any, Tuple
from src.tools.optimizer import HillSaturationModel

class BayesianLiftCalibrator:
    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        
    def calibrate_channel_prior(
        self, 
        prior_mean: float, 
        prior_sd: float, 
        lift_estimate: float, 
        lift_se: float, 
        draws: int = 2000, 
        tune: int = 1000
    ) -> Dict[str, Any]:
        """
        Updates an observational MMM prior with an experimental lift measurement.
        """
        with pm.Model() as model:
            # Prior from Observational MMM (e.g. historical ROAS)
            beta_channel = pm.Normal("beta_channel", mu=prior_mean, sigma=prior_sd)
            
            # Likelihood of the experimental measurement (e.g. from GeoLift)
            observed_lift = pm.Normal(
                "observed_lift", 
                mu=beta_channel, 
                sigma=lift_se, 
                observed=lift_estimate
            )
            
            # MCMC Sampling
            trace = pm.sample(
                draws=draws, 
                tune=tune, 
                return_inferencedata=True, 
                random_seed=self.random_seed,
                progressbar=False
            )
            
        # Extract Posterior Statistics using ArviZ
        # We explicitly suppress standard output logs for clean API execution
        summary = az.summary(trace, var_names=["beta_channel"], round_to=4)
        
        post_mean = float(summary["mean"].iloc[0])
        post_sd = float(summary["sd"].iloc[0])
        
        # HDI limits (High Density Interval) - usually 94% by default in arviz, 
        # but the exact column names might be 'hdi_3%' and 'hdi_97%'
        cols = summary.columns
        hdi_lower = float(summary[cols[2]].iloc[0])
        hdi_upper = float(summary[cols[3]].iloc[0])
        
        # Shrinkage factor calculation
        # How much did the variance shrink from prior to posterior?
        shrinkage_factor = 1.0 - ((post_sd ** 2) / (prior_sd ** 2))
        
        return {
            "prior_mean": prior_mean,
            "prior_sd": prior_sd,
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "hdi_95": [hdi_lower, hdi_upper],
            "shrinkage_factor": float(shrinkage_factor)
        }

class LiftCorrectionEngine:
    @staticmethod
    def calculate_adstock_tail_multiplier(alpha: float, test_duration_weeks: int) -> float:
        """
        Calculates the multiplier to correct for truncated adstock tails.
        If a test runs for T weeks, the cumulative effect captured of a continuous pulse
        under geometric decay alpha is less than the theoretical total effect.
        """
        if alpha < 0 or alpha >= 1:
            raise ValueError("Alpha must be between 0 and 1.")
        
        if test_duration_weeks <= 0:
            raise ValueError("Test duration must be positive.")
            
        # Theoretical total effect of a single pulse: 1 / (1 - alpha)
        total_theoretical_effect = 1.0 / (1.0 - alpha)
        
        # Effect captured within T weeks: sum_{t=0}^{T-1} alpha^t
        captured_effect = (1.0 - (alpha ** test_duration_weeks)) / (1.0 - alpha)
        
        # The multiplier is the ratio of total theoretical effect to captured effect
        multiplier = total_theoretical_effect / captured_effect
        return multiplier

    @staticmethod
    def calculate_saturation_multiplier(
        baseline_spend: float, 
        test_spend: float, 
        v_max: float, 
        k: float, 
        n: float
    ) -> float:
        """
        Calculates the multiplier to map a test performed at a saturated spend level 
        back to the baseline efficiency.
        """
        baseline_aroas = HillSaturationModel.average_roas(baseline_spend, v_max, k, n)
        test_aroas = HillSaturationModel.average_roas(test_spend, v_max, k, n)
        
        if test_aroas <= 0:
            return 1.0 # Fallback if spend is near zero or zero
            
        return baseline_aroas / test_aroas

    @staticmethod
    def apply_corrections(
        raw_lift: float, 
        raw_se: float, 
        tail_multiplier: float = 1.0, 
        saturation_multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """
        Applies multipliers to the raw experimental lift and scales the standard error proportionally.
        """
        total_multiplier = tail_multiplier * saturation_multiplier
        corrected_lift = raw_lift * total_multiplier
        corrected_se = raw_se * total_multiplier
        
        return float(corrected_lift), float(corrected_se)

