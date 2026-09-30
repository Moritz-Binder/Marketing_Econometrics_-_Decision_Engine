from typing import Dict, Any, Optional
from src.agents.state import AgentGraphState
from src.tools.experimentation import ExperimentAuditor, GeoExperimentAuditor
from src.tools.bayesian_calibrator import BayesianLiftCalibrator, LiftCorrectionEngine

class CausalAuditorNode:
    def __init__(
        self, 
        experiment_auditor: ExperimentAuditor, 
        bayesian_calibrator: BayesianLiftCalibrator,
        geo_auditor: Optional[GeoExperimentAuditor] = None
    ) -> None:
        self.experiment_auditor = experiment_auditor
        self.bayesian_calibrator = bayesian_calibrator
        self.geo_auditor = geo_auditor
        
    def __call__(self, state: AgentGraphState) -> Dict[str, Any]:
        intent = state.get("user_intent", "")
        params = state.get("extracted_parameters", {}) or {}
        
        updates: Dict[str, Any] = {}
        
        if intent == "AUDIT_EXPERIMENT":
            experiment_type = params.get("experiment_type", "A/B")
            
            if experiment_type == "Geo" and self.geo_auditor:
                historical_control = params.get("historical_control")
                historical_treatment = params.get("historical_treatment")
                test_duration_days = params.get("test_duration_days", 30)
                
                if historical_control is not None and historical_treatment is not None:
                    drift_dist = self.geo_auditor.simulate_null_drift(
                        historical_control=historical_control,
                        historical_treatment=historical_treatment,
                        test_duration_days=test_duration_days
                    )
                    
                    mde_results = self.geo_auditor.compute_empirical_mde(
                        drift_distribution=drift_dist,
                        alpha=params.get("alpha", 0.05),
                        target_power=params.get("target_power", 0.80)
                    )
                    
                    updates["audit_results"] = {"geo_audit": mde_results}
            else:
                obs_control = params.get("control_conversions", 0)
                obs_variant = params.get("variant_conversions", 0)
                
                # If traffic isn't explicitly provided, fallback to conversions to at least run the ratio check
                control_traffic = params.get("control_traffic", obs_control)
                variant_traffic = params.get("variant_traffic", obs_variant)
                
                if control_traffic > 0 and variant_traffic > 0:
                    srm_results = self.experiment_auditor.audit_sample_ratio_mismatch(
                        observed_control=control_traffic,
                        observed_variant=variant_traffic,
                        expected_control_ratio=params.get("expected_control_ratio", 0.5)
                    )
                    updates["audit_results"] = {"srm_audit": srm_results}
                    
                    # If baseline probabilities are provided, we also compute statistical power
                    if "baseline_p" in params and "relative_lift" in params:
                        power_results = self.experiment_auditor.calculate_statistical_power(
                            sample_size_per_arm=int((control_traffic + variant_traffic) / 2),
                            baseline_p=params["baseline_p"],
                            relative_lift=params["relative_lift"]
                        )
                        updates["audit_results"]["power_audit"] = power_results
                    
        elif intent == "CALIBRATION":
            # Extract channel and raw bayesian parameters
            channel = state.get("target_channels", ["Unknown"])[0] if state.get("target_channels") else "Unknown"
            prior_mean = params.get("prior_mean", 1.0)
            prior_sd = params.get("prior_sd", 0.5)
            raw_lift = params.get("lift_estimate", 1.0)
            raw_se = params.get("lift_se", 0.2)
            
            # Calculate Adstock Tail Corrections if the experiment was truncated (our new engine!)
            alpha = params.get("alpha", 0.0)
            test_duration = params.get("test_duration_weeks", 4)
            
            tail_mult = 1.0
            if alpha > 0 and test_duration > 0:
                tail_mult = LiftCorrectionEngine.calculate_adstock_tail_multiplier(alpha, test_duration)
                
            corrected_lift, corrected_se = LiftCorrectionEngine.apply_corrections(
                raw_lift=raw_lift,
                raw_se=raw_se,
                tail_multiplier=tail_mult,
                saturation_multiplier=1.0 # For simplicity in this node
            )
            
            # Execute PyMC Bayesian Updating
            calib_results = self.bayesian_calibrator.calibrate_channel_prior(
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                lift_estimate=corrected_lift,
                lift_se=corrected_se,
                draws=2000,
                tune=1000
            )
            
            # Append execution metadata
            calib_results["channel"] = channel
            calib_results["corrections_applied"] = {
                "tail_multiplier": tail_mult,
                "original_lift": raw_lift
            }
            updates["calibration_results"] = calib_results

        return updates
